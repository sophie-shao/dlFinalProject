import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt


"""
Here we build the architecture for the residual MLPs that will 
control the fidelity and simplicity of the sketch image.
"""
# builds a transformer style residual block
def _make_block(width: int, dropout_rate: float):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(width, use_bias=False),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Activation(tf.keras.activations.gelu),
        tf.keras.layers.Dropout(dropout_rate),
    ])

# defines the generic MLP backbone that will be used for fidelity and 
# simplicity MLPs
class _BaseMLP(tf.keras.Model):
    def __init__(self, widths=(128,128,128,64,64), dropout_rate=0.1, name="base_mlp"):
        super().__init__(name=name)
        self.blocks = [_make_block(w, dropout_rate) for w in widths]
        self.proj   = tf.keras.layers.Dense(1)
        self.gamma  = tf.Variable(0.1, trainable=True, dtype=tf.float32)

    def call(self, x, training=False):
        for blk in self.blocks:
            h = blk(x, training=training)
            if x.shape[-1] == h.shape[-1]:       # ← safeguard
                x = x + self.gamma * h
            else:
                x = h
        x = self.proj(x)
        return self._scale_and_shift(x)


class FidelityMLP(_BaseMLP):
    # maps (0...1) to fidelity in [10, 110]

    def _scale_and_shift(self, x):
        return tf.sigmoid(x) * 100.0 + 10.0


class SimplicityMLP(_BaseMLP):
    # maps (0...1) to simplicity in [0, 0.35]

    def _scale_and_shift(self, x):
        return tf.sigmoid(x) * 0.35


"""
The next four functions load_image, get_saliency_map, make_pencil_sketch,
and vectorize_strokes are helper functions, that we will use in our 
training loop.
"""
# load image helper function
def load_image(path, size=(512, 512)):
    img = Image.open(path).convert("RGB").resize(size)
    return np.asarray(img, dtype=np.float32) / 255.0

# make saliency map 
def get_saliency_map(image_np):
    gray = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # compute image gradients in horizontal and vertical directions
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # combine
    saliency_map = np.abs(dx) + np.abs(dy)

    # normalize to (0,1)
    saliency_map /= saliency_map.max() + 1e-8

    return saliency_map


# use saliency map as a mask and turn image into a pencil sketch
def make_pencil_sketch(image, saliency, simp_thr, sigma_s):
    # convert from rgb to opencv bgr format
    bgr = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # opencv's built in pencilSketch filter
    # sigma_s: spatial smoothing (how soft strokes are) - we are treating our 
    # fidelity value as this
    # sigma_r: range smoothing (how much variation to preserve)
    # shade_factor: how dark shading/background is
    grayscale_sketch, _ = cv2.pencilSketch(bgr, sigma_s=sigma_s, sigma_r=0.07, shade_factor=0.05)

    # invert colors
    pencil = 1.0 - grayscale_sketch.astype(np.float32) / 255.0

    # use saliency map as mask to choose where to keep pixels
    mask = (saliency > simp_thr).astype(np.float32)

    # apply mask and return
    return pencil * mask

# vectorize the strokes on the pencil sketch image
def vectorize_strokes(pencil_img, min_length, sample_every, jitter_px, widths):
    """
    inputs:
    - pencil_img (2D float array): nonzero pixels are the penci; strokes
    - min_length (int): minimum number of points in a contour to keep it
    - sample_every (int): stride for downsampling each contour's points
    - jitter_px (double): maximum random offset to apply to sampled points
    - widths (set): a tuple of possible line widths to randomly choose from
    """
    H, W = pencil_img.shape

    # threshold and dilate (fills gaps to make sure strokes are connected) the pencil strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.dilate((pencil_img > 0.1).astype(np.uint8), kernel, 1)

    # skeletonize to thin each pencil stroke
    skel = skeletonize(binary).astype(np.uint8)
    # 8 bit mask for contour detection
    mask8 = (skel * 255).astype(np.uint8)

    cnts_tuple = cv2.findContours(mask8, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cnts = cnts_tuple[-2]

    # initialize canvas
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)

    # draw each contour
    for c in cnts:
        pts = c.squeeze()
        # skip contours that aren't long enough or shape weird
        if pts.ndim != 2 or len(pts) < min_length:
            continue

        # downsample
        pts_ds = pts[::sample_every]
        if len(pts_ds) == 0:
            continue

        # randomly jitter to make it more hand drawn style within image bounds
        jitter = np.random.uniform(-jitter_px, jitter_px, pts_ds.shape)
        pts_j = np.clip(pts_ds + jitter, 0, [W - 1, H - 1])

        # redraw thees strokes as vector lines of varying width onto canvas
        poly = [tuple(p) for p in pts_j.astype(int)]
        if len(poly) < 2:
            continue
        w = np.random.choice(widths)
        draw.line(poly, fill="black", width=w)

    return np.asarray(canvas, dtype=np.float32) / 255.0


"""
The functions below define the grid generation, training loop, and main driver
"""
# creates the grid of sketch images with varying values of fidelity and simplicity
def generate_grid(
    path,
    fidelity_values,
    simplicity_values,
    fidelity_mlp,
    simplicity_mlp,
):
    img = load_image(path)
    sal = get_saliency_map(img)

    # initialize grid
    grid = []

    # fidelity and simplicity values
    params_base = [
        (5, 1, 0.2, (1, 2)),   # high fidelity
        (15, 3, 0.5, (1, 2)),  # medium
        (30, 6, 1.0, (2, 3)),  # low
    ]

    fid_tensor = tf.constant([[v] for v in fidelity_values], dtype=tf.float32)
    simp_tensor = tf.constant([[v] for v in simplicity_values], dtype=tf.float32)

    fid_sigmas = fidelity_mlp(fid_tensor).numpy().flatten()
    simp_thresholds = simplicity_mlp(simp_tensor).numpy().flatten()

    print("fidelity values from FidelityMLP :", fid_sigmas)
    print("simplicity values from SimplicityMLP:", simp_thresholds)

    # for each box in the grid generate the pencil image
    for i, sigma in enumerate(fid_sigmas):
        row = []
        param_idx = min(int(sigma / 40), 2)
        min_len, samp, jit, w = params_base[param_idx]

        for thresh in simp_thresholds:
            pencil = make_pencil_sketch(img, sal, thresh, sigma)
            vec = (
                vectorize_strokes(pencil, min_len, samp, jit, w)
                if pencil.sum() > 0
                else np.ones_like(pencil)
            )
            row.append(vec)
        grid.append(row)
    return img, grid, fid_sigmas, simp_thresholds

# training loop to minimize loss between inputs and targets, where 
# fidelity inputs are (0, 1) and target is [10, 110]
# simplicity inputs are (0, 1) and target is [0, 0.35]
def pretrain_mlp(mlp, inputs, targets, *, epochs=1000, lr=0.005):
    opt = tf.keras.optimizers.Adam(lr)
    loss_fn = tf.keras.losses.MeanSquaredError()
    x = tf.constant([[v] for v in inputs], dtype=tf.float32)
    y = tf.constant([[v] for v in targets], dtype=tf.float32)

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            pred = mlp(x, training=True)
            loss = loss_fn(y, pred)
        opt.apply_gradients(zip(tape.gradient(loss, mlp.trainable_variables),
                                mlp.trainable_variables))
        if (epoch + 1) % 250 == 0:
            print(f"[{mlp.name}] epoch {epoch+1:4d}/{epochs}  loss={loss.numpy():.4f}")

# generates sketch from image using the helper functions defined previously
def generate_sketch(path, f_val, s_val, fid_mlp, simp_mlp):
    img = load_image(path)
    sal = get_saliency_map(img)

    # fidelity value 
    sigma = fid_mlp(tf.constant([[f_val]], tf.float32)).numpy().item()
     # simplicity threshold
    thresh = simp_mlp(tf.constant([[s_val]], tf.float32)).numpy().item()

    # jittering
    min_len, samp, jit, widths = (
        (5, 1, 0.2, (1, 2))
        if sigma < 40
        else (15, 3, 0.5, (1, 2))
        if sigma < 80
        else (30, 6, 1.0, (2, 3))
    )

    # create the pencil sketch
    pencil = make_pencil_sketch(img, sal, thresh, sigma)

    # vectorize the pencil strokes
    vec = (
        vectorize_strokes(pencil, min_len, samp, jit, widths)
        if pencil.sum() > 0
        else np.ones_like(pencil)
    )
    return vec, sigma, thresh

# uses GPU if availible (only Sophie has GPU)
def set_tensorflow_device():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"Using GPU ({len(gpus)}x)")
        return "GPU"
    print("Using CPU")
    return "CPU"


def main():
    set_tensorflow_device()

    # create MLP instances
    fidelity_mlp = FidelityMLP(name="fidelity_mlp")
    simplicity_mlp = SimplicityMLP(name="simplicity_mlp")

    # build once (TensorFlow requirement)
    _ = fidelity_mlp(tf.ones((1, 1), tf.float32))
    _ = simplicity_mlp(tf.ones((1, 1), tf.float32))

    # quick pre-training to give sensible starting mapping
    pretrain_mlp(fidelity_mlp, [0.0, 0.5, 1.0], [30.0, 60.0, 90.0])
    pretrain_mlp(simplicity_mlp, [0.0, 0.5, 1.0], [0.05, 0.2, 0.3])

    # Create output directory if it doesn't exist
    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)

    # Process all images in the "in" folder
    in_dir = "in"
    if not os.path.exists(in_dir):
        raise FileNotFoundError(f"Input directory '{in_dir}' not found")

    # Get all image files in the input directory
    valid_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    
    for filename in os.listdir(in_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            image_files.append(os.path.join(in_dir, filename))
    
    if not image_files:
        print(f"No images found in '{in_dir}' directory")
        return
        
    print(f"Found {len(image_files)} images to process")
    
    # Set fidelity and simplicity values for the 3x3 grid
    f_vals = [0.0, 0.5, 1.0]
    s_vals = [0.0, 0.5, 1.0]
    
    # Process each image
    for i, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)
        basename, ext = os.path.splitext(filename)
        output_path = os.path.join(out_dir, f"{basename}_grid.png")
        
        print(f"Processing image {i+1}/{len(image_files)}: {filename}")
        
        try:
            # generate the sketch grid for this image
            img, sketch_grid, sigmas, thresholds = generate_grid(img_path, f_vals, s_vals, fidelity_mlp, simplicity_mlp)
            
            # plot grid
            fig, axes = plt.subplots(3, 3, figsize=(11, 11))
            for i in range(3):
                for j in range(3):
                    axes[i, j].imshow(sketch_grid[i][j])
                    axes[i, j].axis("off")
                    axes[i, j].set_title(
                        f"F={f_vals[i]:.1f} → sigma={sigmas[i]:.1f}\n"
                        f"S={s_vals[j]:.1f} → thres={thresholds[j]:.3f}",
                        fontsize=9,
                    )
            plt.tight_layout()
            plt.savefig(output_path, dpi=120)
            plt.close()
            
            print(f"Saved grid to {output_path}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    print("Everything is saved")

if __name__ == "__main__":
    main()
