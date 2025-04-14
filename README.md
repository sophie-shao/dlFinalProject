# dlFinalProject

### Title: Sketchy Business

### Members: 
Apoorva Talwalkar (`atalwalk`), Sophie Shao (`swshao`), Fiona Fan (`ffan7`)

## Introduction:

We want to be able to convert images into lofi sketches by implementing the existing paper [CLIPascene](https://arxiv.org/pdf/2211.17256) 

* The paper's objective is to ...
* We chose this paper because
* The type of problem is computer vision

## Related Work:

* Related works include CLIPasso, which focused on single object sketching, while CLIPascene extends this to complex scenes by separating foreground and background elements. CLIPasso converts images of objects into abstract sketches by leveraging the semantic understanding capabilities of CLIP (Contrastive Language-Image Pretraining). The degree of abstraction in CLIPasso is controlled by varying the number of strokes used to create the sketch
* Public implementations:
  1. https://github.com/yael-vinker/SceneSketch

## Data:

* 

## Metholodology:

## Metrics:

## Ethics:
The broader societal implications of this paper could include friction with artists or designers, whose style or work may include hand converting images into sketches. In the past month, with ChatGPT's update for generative images, there has been a lot of backlash for Artificially Generated Images. Specifically, many artists have issue with the model imitating a real artists's style without their consent (eg. the Studio Ghibli trend). Artists generally have problems with this because they view their style are part of their Intellectual Property and part of their identity. This project, unlike generated images, avoids the potetial issues in this sphere because it just outputs a black and white sketch, not a fully rendered or finished image and not in any particular style. It would be hard for any artist or person to claim that a black and white sketch is their property or style.

The major stakeholders of this problem are us (the creators of the algorithm) and users of the program. There aren't many extreme consequences to a mistake in this program. If the model learns incorrectly and outputs a "bad" sketch, the user can simply rerun the algorithm to try and generate a better sketch or stop using the algorithm if it doesn't work consistently. Since this model doesn't try and mimic a style, there is no harm to artists.

## Division of Labor:
