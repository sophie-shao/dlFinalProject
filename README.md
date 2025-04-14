# dlFinalProject

### Title: Sketchy Business

### Members: 
Apoorva Talwalkar (`atalwalk`), Sophie Shao (`swshao`), Fiona Fan (`ffan7`)

## Introduction:

We want to be able to convert images into lofi sketches by implementing the existing paper [CLIPascene](https://arxiv.org/pdf/2211.17256).

* CLIPascene's objective is to **convert an input image into a customizable sketch** based off of two parameters of abstraction-- fidelity and visual simplicity. Fidelity determines how closely the output sketch matches the original input image and visual simplicity controls how sparse the sketch is.
* We chose this paper because it provides a **straightforward approach to transforming images into a new artistic style** while giving users some adjustability. Its methods also form the foundation for more advanced projects, for instance NeuralSVG, which means we can build on it later to tackle more complex transformations. We think it’s especially **valuable for artists and designers** because it helps them explore ideas, create visual drafts, and speed up the creative process.
* The type of problem is **generative computer vision**

## Related Work:

* Related works include CLIPasso, which focused on single object sketching, while CLIPascene extends this to complex scenes by separating foreground and background elements. CLIPasso converts images of objects into abstract sketches by leveraging the semantic understanding capabilities of CLIP (Contrastive Language-Image Pretraining). The degree of abstraction in CLIPasso is controlled by varying the number of strokes used to create the sketch and generates sketches that can effectively capture both the semantics and structure of subjects like flamingos and horses. However, the performance decreases for images with backgrounds, especially at higher abstraction levels. This is where CLIPascene builds off from.
* Public implementations:
  1. https://github.com/yael-vinker/SceneSketch

## Data:

* 

## Metholodology:
For this project, we will need to train 2 MLPs. The MLP receives an initial set of control points and returns a vector of offsets with respects to the initial stroke locations. For a given n levels of fidelity and m levels of simplicity, we first construct a sketch abstraction matrix of size m * n . We name one axis the fidelity axis and the other the simplicity axis. We first produce a set of sketches along the fidelity axis which fills in the first row of our matrix. Then for each of these sketches, we perform an iterative simplification to complete each column of the matrix. We train a second MLP for this task. This second MLP receives a random-valued vector and learns an n-dimensional vector representing the probability of the i-th stroke appearing in the final sketch and outputs a simplified sketch. We think the hardest part of this project will be successfully training the MLPs to work in conjunction to generate good outputs.

## Metrics:
We plan to test our model on various images with different backgrounds and foregrounds. Success for this project means generating recognizable sketches that capture the essence of the input scene and producing distinct and meaningful variations along both abstraction axes. Accuracy will be measured both quantitatively and qualitatively. Quantitatively, accuracy is determined by using CLIP similarity scores between sketches and original images. Qualitatively, we can compare the different levels of abstraction by eye. The authors hoped to replicate the images with a sketch and quantified their results by comparing different levels of abstraction to each other and the original image and also comparing their outputs to other methods like CLIPasso. 

Base goal: successfully implement the basic CLIPascene pipeline that can generate sketches for simple scenes
Target goal: successfully separating the background from foreground, achieving comparable results to the paper across both abstraction axes for a variety of scenes
Stretch goal:  extend the method with additional controls or optimizations, such as style transfer

## Ethics:
The broader societal implications of this paper could include **friction with artists or designers**, whose style or work may include hand converting images into sketches. In the past month, with ChatGPT's update for generative images, there has been a lot of backlash for Artificially Generated Images. Specifically, **many artists have issue with the model imitating a real artists's style without their consent** (eg. the Studio Ghibli trend). Artists generally have problems with this because they view their style are part of their Intellectual Property and part of their identity. This project, unlike generated images, **avoids the potetial issues in this sphere because it just outputs a black and white sketch**, not a fully rendered or finished image and not in any particular style. It would be hard for any artist or person to claim that a black and white sketch is their property or style.

The major stakeholders of this problem are us (the creators of the algorithm) and users of the program. There aren't many extreme consequences to a mistake in this program. If the model learns incorrectly and outputs a "bad" sketch, the user can simply rerun the algorithm to try and generate a better sketch or stop using the algorithm if it doesn't work consistently. Since this model doesn't try and mimic a style, there is no harm to artists.

## Division of Labor:
