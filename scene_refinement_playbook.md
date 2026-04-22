# ManimGL Scene Refinement Playbook

## Purpose

This file is the working template for refining the ManimGL scene files in this repo.

Its job is to connect three layers:

1. `blogpost_content.txt` gives the narrative purpose and research/story arc.
2. `scene_description.txt` gives the intended visual beats for each scene.
3. The stronger scene files show how those beats are translated into clean, readable, renderable ManimGL code.

This playbook is intentionally living documentation. As we refine more scenes, we should keep appending lessons here instead of re-learning the same patterns from scratch.

## Canonical Reference Files

These are the current "best examples" to imitate:

| File | Why it is a strong reference |
| --- | --- |
| `clip_shared_embedding_space_sc1.py` | Clear left-to-right narrative pipeline, strong object continuity, good use of transforms to explain CLIP end to end. |
| `clip_similarity_matrix_sc2.py` | Good example of turning an abstract computation into stepwise visual logic with a grid, vectors, dot products, and emphasis beats. |
| `umap_visualization_scene_sc4.py` | Best example of multi-stage scene architecture, camera choreography, seeded synthetic data, and persistent state passed across methods. |
| `mlp_decision_boundary_scene_sc5.py` | Best example of pedagogical abstraction: it does not simulate literal training, but it communicates model behavior cleanly and convincingly. |

## Core Translation Rule

When refining a scene, always translate in this order:

1. `blogpost_content.txt`: What claim is the scene helping the viewer believe?
2. `scene_description.txt`: What must appear on screen for that claim to land?
3. Code: What is the simplest sequence of reusable objects and transforms that makes the idea obvious?

The refined files consistently optimize for understanding, not for literal realism.

That means:

- simplify math visuals when the exact implementation would be visually noisy
- use toy values when they make the mechanism legible
- fake intermediate geometry if it explains the concept honestly
- preserve the truth of the idea, not the full complexity of the underlying model

## What The Good Files Have In Common

### 1. They are beat-driven, not object-driven

The stronger files are organized around story beats.

A good beat usually does one thing:

- introduce objects
- transform them
- highlight a contrast
- compress the idea into a takeaway

The code should mirror that progression. If a scene has many beats, split them into stage methods. If it only has a single pipeline, one `construct()` with local helpers can still work well.

### 2. They preserve visual continuity

The strongest pattern across the files is continuity through transformation.

Instead of fading one idea out and showing a totally new one, the code often does this:

- text becomes tokens
- tokens shrink into an encoder
- the encoder emits a vector
- the vector becomes a point
- the point becomes part of a cluster

This is why `ReplacementTransform`, `Transform`, and moving copies of existing objects are so important in these files. The viewer should feel that each new object came from the previous one.

### 3. They keep one semantic color mapping per scene

The refined files use color semantically, not decoratively.

Examples:

- text path vs image path use distinct accents in Scene 1
- matched vs mismatched matrix cells use warm vs cool contrast in Scene 2
- semantic categories use stable colors in Scene 4
- class A vs class B keep the same colors through the full MLP boundary sequence in Scene 5

Rule of thumb:

- neutrals for scaffolding and layout
- 2 to 4 accent colors for meaning
- once a color means something, keep it stable for the whole scene

### 4. They use simplified but deterministic data

The refined files almost never depend on heavy real data pipelines during rendering.

Instead they use:

- seeded random point generation
- hand-crafted toy vectors
- synthetic clusters
- procedurally defined curves

This gives three benefits:

1. the render is stable
2. the scene is easy to tune visually
3. the animation remains about explanation, not data wrangling

If randomness is used, seed it unless variation is explicitly desirable.

### 5. They are performance-aware

The better files regularly choose "enough to convey the idea" instead of brute-force detail.

Examples:

- Scene 1 animates only a subset of image patches flying into the encoder instead of every patch.
- Scene 4 uses many points, but still keeps them lightweight and generated procedurally.
- Scene 5 creates a convincing boundary-sharpening effect without actually training anything frame by frame.

Refinement should always ask:

"What is the minimum amount of geometry/motion needed for the viewer to understand this beat?"

### 6. They keep styling reusable and localized

There are two recurring styles of code organization:

#### Pattern A: Single-scene local helper pattern

Use this when the scene is basically one continuous pipeline.

Characteristics:

- local helper functions inside `construct()`
- layout constants near the top of `construct()`
- one long but readable sequence of beats

Best current example:

- `clip_shared_embedding_space_sc1.py`

#### Pattern B: Multi-stage scene class pattern

Use this when the scene has multiple conceptual phases, camera changes, or reused state.

Characteristics:

- module-level palette/constants
- module-level helper functions
- `construct()` only orchestrates stage order
- `self._...` attributes carry state from one stage to the next

Best current examples:

- `umap_visualization_scene_sc4.py`
- `mlp_decision_boundary_scene_sc5.py`

## File-By-File Lessons

### `clip_shared_embedding_space_sc1.py`

What it teaches:

- Start with a direct visual metaphor and commit to it.
- Keep the whole scene centered around a single pipeline: text branch + image branch -> shared space.
- Build custom helper constructors for repeated motifs: encoder boxes, token blocks, numeric vectors, glow dots.
- Use transforms to keep the story continuous from symbolic input to embedding space.
- End with a payoff frame that generalizes beyond the single example.

Patterns worth reusing:

- scene-specific helpers defined close to their only usage
- branch symmetry in layout
- final zoom into the conceptual center of the scene
- use of an ending cluster tableau to summarize the mechanism

Important refinement principle from this file:

When a scene explains a mechanism, first show one concrete example clearly, then widen to the more general pattern.

### `clip_similarity_matrix_sc2.py`

What it teaches:

- Abstract computations become much clearer when the layout is fixed before the arithmetic starts.
- A grid gives the viewer a stable frame of reference.
- Reusing visible vectors for each cell computation makes the math feel causal instead of magical.
- Emphasis beats matter: dim off-diagonal, pop diagonal, then reverse the emphasis.

Patterns worth reusing:

- create static axes first, then animate computation into them
- use copies of canonical objects instead of re-creating new unexplained objects
- encode logic through contrast, opacity, and scale pulses
- replace values in place once a new interpretation is introduced

One practical caveat:

The class docstring says the scene goes on to `L_image`, `L_text`, and `L_CLIP`, but the current implementation stops after row-wise normalization. For future refinement work, keep comments, docstrings, and scene descriptions honest about what is already implemented vs still planned.

### `umap_visualization_scene_sc4.py`

What it teaches:

- Complex scenes benefit from a stage-method structure.
- Camera motion can be part of the explanation, not just decoration.
- You can reuse the same cloud through multiple conceptual views instead of rebuilding everything from scratch.
- The `self._stored_state` pattern makes cross-stage choreography much easier.

Patterns worth reusing:

- palette constants at module scope
- `title_text()` and `caption_text()` helpers for consistent typography
- stage methods that each do one conceptual job
- storing the current visual state on `self` after every major transition
- seeded point generation for repeatable cluster layouts

Important refinement principle from this file:

When the concept is about changing viewpoint or projection, let the camera and object transformation do the teaching together.

### `mlp_decision_boundary_scene_sc5.py`

What it teaches:

- Good explanatory animation often uses pedagogical approximations instead of literal internals.
- A side diagram can support the main plot without stealing focus.
- Layer-by-layer transformation works well when each stage has a clearly visible visual consequence.
- `ValueTracker` plus `always_redraw` is a clean way to add "training progression" without overcomplicating the scene.

Patterns worth reusing:

- build a base plot once, then keep reusing it
- highlight one layer/group at a time when explaining model internals
- use synthetic transforms to communicate feature warping
- return to the original space before showing the final nonlinear boundary
- finish on a polished end state with labels and a stable frame

Important refinement principle from this file:

If the real process is too opaque to animate directly, animate an equivalent explanatory metaphor that preserves the core claim.

## The Shared Storytelling Formula

Across the better files, the animation logic often follows this structure:

1. Establish a stable frame
2. Introduce the main objects
3. Show one transformation at a time
4. Highlight the contrast or mechanism
5. Compress the result into a summary visual
6. Land on a final frame that states the takeaway

This is a strong default template for the remaining scenes.

## Convergence Scene Lessons

### What went wrong

- A trend chart with nonlinear spacing can become geometrically cleaner but conceptually confusing if the axis still reads like a normal quantitative axis.
- Replacing that chart with a boxed infographic drifted too far from the original storytelling intent. It made the scene feel stylistically off relative to the repo and introduced a decorative frame that was not carrying explanatory value.
- A verifier that only checks "no overlap" will approve frames that still feel misleading, awkward, or editorially wrong.

### What went right

- The chart-specific verification spec was a useful direction. It caught issues that a generic overlap checker missed:
  - axis label vs axis collisions
  - arrow paths running through their own labels
  - annotations crowding a curve
  - dense tick labeling on a nonlinear x-axis
- Using the last rendered frame as a critique surface was useful, but only when paired with explicit semantic checks. Pure visual inspection is too ad hoc to be the only guardrail.

### Lesson learned

- Do not "escape" a difficult chart by switching to a different visual metaphor unless the scene description or blogpost logic truly supports that change.
- Prefer staying inside the original visual language and refining:
  - axis honesty
  - label hierarchy
  - annotation placement
  - semantic clarity
- Use stronger scene-type-specific verification to catch local readability failures, but keep editorial judgment for bigger questions like "should this still be a chart at all?"

### Practical rule for future refinements

- When a scene starts as a chart, first try:
  1. fewer ticks
  2. better label placement
  3. explicit scale disclosure
  4. chart-aware verifier rules

- Only consider changing the visual metaphor after those steps fail and after checking that the alternative still matches the original narrative intent.

## Recommended Code Template For New Or Refined Scenes

Choose one of these two skeletons.

### Template A: Small linear scene

Use for scenes like "input -> process -> output".

```python
from manimlib import *
import numpy as np


class ExampleScene(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        # Scene-local helpers because they are not reused elsewhere
        def make_label(text, color=WHITE, font_size=28):
            return Text(text, font_size=font_size, color=color)

        def make_node(width=2.0, height=1.2, color=GREY_B):
            box = RoundedRectangle(width=width, height=height, corner_radius=0.12)
            box.set_stroke(color, width=1.5)
            box.set_fill(GREY_E, opacity=0.25)
            return box

        # Beat 1: establish inputs
        # Beat 2: move inputs into process
        # Beat 3: transform into outputs
        # Beat 4: zoom/summary/payoff
```

### Template B: Multi-stage conceptual scene

Use for scenes with multiple phases, camera changes, or reused state.

```python
from manimlib import *
import numpy as np


BG_COLOR = "#0f1117"
ACCENT_A = BLUE
ACCENT_B = ORANGE


def title_text(s, scale=0.7, color=WHITE):
    t = Text(s, color=color)
    t.scale(scale)
    return t


class ExampleConceptScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        self.camera.frame.set_width(14)

        self.stage1_intro()
        self.stage2_mechanism()
        self.stage3_comparison()
        self.stage4_payoff()

    def stage1_intro(self):
        # Build the starting objects
        # Save anything needed later on self._...
        pass

    def stage2_mechanism(self):
        # Transform existing state rather than resetting from scratch
        pass

    def stage3_comparison(self):
        # Show contrast, failure mode, or alternate view
        pass

    def stage4_payoff(self):
        # Land on the clearest takeaway frame
        pass
```

## Refinement Workflow For The Remaining Scene Files

When refining another file in this repo, follow this sequence:

1. Read the corresponding scene note in `scene_description.txt`.
2. Write the scene claim in one sentence.
3. Break the scene into 3 to 6 beats max.
4. Decide whether this should be a single-pipeline scene or a multi-stage scene.
5. Define the visual vocabulary before animating:
   - palette
   - titles/captions
   - reusable shapes
   - coordinate system / anchors
6. Block the static layout first.
7. Add transitions that preserve continuity.
8. Add emphasis beats:
   - opacity changes
   - color contrast
   - scale pops
   - temporary labels
9. Simplify anything that feels too literal or too busy.
10. End on a frame that could be screenshotted and still communicate the key idea.

## Practical Heuristics

### Use local helpers when:

- the shape is specific to one file
- the scene is short and linear
- moving helper code to module scope would make the file harder to read

### Use module-level helpers when:

- multiple stages share the same constructors
- the scene has many methods
- the scene has a reusable palette/typography system

### Use `self._state` when:

- later stages need objects created earlier
- you want to transform the same mobjects over time
- you want the scene to feel continuous across stages

### Prefer transforms over replacement by fade when:

- the viewer should understand causal continuity
- an output conceptually comes from an input
- you are explaining a pipeline or mechanism

### Prefer fresh fade-ins when:

- the new object is a comparison view, not a transformation
- the previous object would create visual clutter
- the concept needs a reset in viewer attention

## Quality Checklist

Before calling a scene "refined", check the following:

- The scene has a single clear claim.
- The viewer can follow the scene without reading a lot of text.
- Each beat introduces only one new idea.
- Colors have stable meanings.
- Layout anchors stay stable unless movement itself is explanatory.
- Randomness is seeded when reproducibility matters.
- The motion density is manageable.
- The final frame lands on the takeaway.
- The code structure matches scene complexity.
- Comments/docstrings match the actual implementation state.

## Common Anti-Patterns To Avoid

- Do not make weak-separation scenes look perfectly separated.
- Do not overload the frame with too many labels at once.
- Do not use long text paragraphs to explain what animation should show visually.
- Do not rebuild entire scenes from scratch when a transform would tell the story better.
- Do not mix too many unrelated accent colors.
- Do not leave style constants scattered throughout the file.
- Do not use unseeded randomness if the final image composition matters.
- Do not let comments promise stages that the code does not yet implement.

## How This Connects To Repo Workflow

Use this file together with:

- `scene_description.txt` for scene intent
- `blogpost_content.txt` for narrative alignment
- `README.md` for environment/render commands

This playbook should stay focused on scene design, code structure, and refinement choices rather than installation details.

## Lessons Learned Log

Append future observations here as we refine more files.

### Template for new entries

```md
## Lesson: <short title>

Date:

Scene/file:

What changed:

Why it worked:

What to reuse next time:

What to avoid next time:
```

## Lesson: Show the aggregation step when a metric compresses many distances into one scalar

Date: 2026-04-22

Scene/file: `silhouette_weak_separation_scene.py`

What changed:

The scene stopped treating the blue and orange connectors as self-explanatory bundles. Instead, it now transforms copies of those exact displayed distances into small summary bars and labels their means as `a` and `b` before landing the silhouette score on the number line.

Why it worked:

Without the aggregation step, viewers can understand that two kinds of distances are being compared but still fail to understand what the symbols mean. Showing the set of measurements, then the averaging step, then the scalar makes the metric feel causal instead of decorative.

What to reuse next time:

For scenes that collapse many local measurements into one statistic, preserve this chain on screen:
- measured examples
- grouped summary of those same examples
- aggregation label such as mean / sum / max
- final scalar payoff

What to avoid next time:

Do not jump directly from several colored connectors to symbolic variables if the viewer still has to infer how those measurements became the scalar.

### Current starter lessons

#### Lesson: Use layout verifiers for annotation-heavy scenes

LLM-generated Manim code is especially fragile around formulas, captions, axes, and lower-third stacks. Add geometric verifier checks for overlap, minimum vertical gap, and frame bounds before animating those groups so layout problems fail fast instead of only showing up after render.

#### Lesson: Keep the visual metaphor stable

Once a scene chooses a metaphor such as pipeline, grid, scatter plot, or network, most later beats should evolve from that same metaphor instead of replacing it with a disconnected one.

#### Lesson: The clearest scenes earn their abstraction

The best files are selective about realism. They show just enough of the mechanism for the idea to feel true and intuitive.

#### Lesson: Endings matter more than extra intermediate detail

A strong final frame often contributes more to comprehension than one more busy sub-step in the middle.

#### Lesson: Split bundled scene files before polishing

When a single Python file contains multiple unrelated scenes, refinement gets slower and reasoning gets muddier. Split the file into one canonical file per scene first, then refine each scene independently. This reduces naming confusion, avoids duplicate exports, and makes per-scene verification much easier.

#### Lesson: `self.embed()` is a late-stage tool, not part of the default workflow

Interactive embeds are helpful for final human tuning, but they slow down automated iteration. The default workflow should assume:
- automatic verifier first
- last-frame render second
- human interactive inspection only near the end

#### Lesson: Keep the renderer simple; improve the scene and verifier instead

When a scene feels broken, the first response should not be adding more wrapper layers around `manimgl`. Prefer:
- fixing the scene layout
- improving the semantic verifier
- clarifying the render command

#### Lesson: If a viewer asks "what is that shape for?", the primitive is under-explained

This came up directly in the L2 normalization scene. A unit circle or sphere may be mathematically correct, but it does not automatically read as meaningful to a viewer. When a primitive is central to the explanation, make its role explicit on screen:
- name it in plain language
- say what property it represents
- show why the animated object relates to it

In practice:
- a circle should read as `unit circle`, not just as a decorative boundary
- a vector outside the circle should read as `before normalization`, not just as a random oversized arrow
- faint dots on a sphere should either be explained as `other normalized embeddings` or removed

#### Lesson: Legends should earn their space by naming roles, not symbols

A detached legend with labels like `u` and `v` often helps the code author more than the viewer. Prefer legends or captions that explain semantic roles:
- `anchor direction`
- `comparison direction`
- `matched pair`
- `mismatched pair`

If the label only names an object but does not clarify its purpose, it is usually noise.

#### Lesson: Camera guidance should be documented as a principle, not a magic number

For 3D explanatory scenes, the reusable rule is not `always use theta = 85`. The reusable rule is:
- flatten irrelevant depth
- zoom until the key relationship dominates the frame
- choose the camera that makes the concept easiest to read

For angle-comparison scenes, that usually means a more horizontal and tighter view than the default dramatic 3D angle.

Extra runtime abstraction can easily become more complex than the original problem.

#### Lesson: Deterministic verification should be scene-type aware

Generic overlap checks are necessary but not sufficient. The more useful verifier pattern is:
- tag semantic roles in the scene
- run scene-type-specific checks
- then inspect the rendered frame

For chart scenes in particular, useful checks include:
- axis label vs axis collisions
- label vs arrow collisions
- label vs curve clearance
- nonlinear scale disclosure
- dense tick warnings on nonlinear axes

#### Lesson: Use the rendered frame as a critic, not as the only verifier

The last rendered frame is valuable because it reveals "geometrically valid but visually illogical" compositions. But visual review should sit on top of structured checks, not replace them. The stronger workflow is:

1. object-level deterministic checks
2. render last frame
3. visual critique of the frame
4. final human review

#### Lesson: Do not change the visual metaphor too quickly

If a scene begins as a chart, pipeline, scatter plot, or architecture diagram, try to repair that metaphor before switching to a different one. A new metaphor can solve one local problem while breaking narrative alignment, repo consistency, or viewer expectation.

#### Lesson: Remove incidental detail when the core teaching beat is simple

Scene 3 worked better after dropping the extra image/text panel and rebuilding around the actual conceptual beats:
- raw vector
- normalized vector
- sphere of directions
- angle between two vectors
- cosine changing with angle

If the scene claim is simple, extra representational baggage can make the result harder to read instead of richer.

#### Lesson: Every geometric primitive must earn its explanation

If a scene shows a circle, sphere, arc, or set of points, the viewer should be able to answer "what is this object doing here?" within a second or two.

For Scene 3 this means:
- the circle must clearly read as the unit circle
- the long vector must clearly read as an unnormalized vector whose magnitude is too large
- the sphere must clearly read as the set of all unit-length directions
- any extra dots on the sphere must either support the explanation or be removed

If an object cannot be justified quickly, it is probably noise rather than scaffolding.

#### Lesson: Legends and labels should explain semantics, not just names

A label like `u` or `v` is too weak if the viewer does not already know what those symbols stand for. Prefer labels that communicate role, for example:
- anchor vector
- comparison vector
- image embedding
- text embedding

The symbol can still be present, but the role should be visually obvious.

#### Lesson: In 3D explanatory scenes, camera choice should serve the concept, not the drama

Specific camera values such as `theta = 85` are not the reusable lesson by themselves. The reusable lesson is:
- choose the camera angle that makes the key relation easiest to read
- reduce depth ambiguity when the scene is about angle, alignment, or comparison
- zoom in enough that the important vectors and angle marker dominate the frame

For angle-based similarity scenes, a more horizontal and closer camera often helps because it makes the angle cue more legible and the viewer spends less effort decoding the 3D projection.

#### Lesson: In 3D scenes, fixed-frame overlays need stronger hierarchy than the 3D objects

For 3D explanatory scenes, the viewer can easily lose the narrative if all labels live inside the 3D world. Titles, formulas, captions, and numeric readouts usually work better as fixed-in-frame overlays with:
- a clear top stack
- a clear bottom stack
- minimal left/right legend content

#### Lesson: Avoid creating text objects inside `always_redraw`

`always_redraw` is fine for geometry, arcs, and moving markers, but repeatedly recreating text inside it is brittle and slow. Prefer:
- create the text once
- move it with an updater

This is especially important in our environment because SVG text generation can be flaky.

#### Lesson: For MMD-style distribution scenes, prefer soft neighborhood weighting over explicit point-pair links

Scene/file:
- `mmd_distribution_difference_scene.py`

Approved:
- keep class colors semantically stable across the whole scene
- scale the current anchor point so the viewer knows what is being probed
- use a kernel ring around the anchor to imply local support
- show kernel contribution with soft opacity weighting on nearby points
- step through the three terms sequentially:
  - within group A
  - within group B
  - across groups
- collapse those terms into a compact summary view after the viewer understands the mechanism

Not approved:
- long crisscrossing lines between selected dots when the concept is MMD
- visuals that read like nearest-neighbor matching, transport, or graph edges
- relying on a few arbitrary point-to-point links as the main explanatory metaphor for a distribution-level statistic

Why:
- MMD is better taught as an average of soft kernel similarities than as a set of explicit pairings
- pairwise lines pull viewer attention toward individual geometry instead of the distribution-level comparison

Reusable rule:
- if the statistic is based on soft similarity, the animation should usually look like weighting, fading, or local influence, not matching

#### Lesson: Reserve a dedicated text lane for scatter-based scenes and use clean text swaps

Scene/file:
- `mmd_distribution_difference_scene.py`

Approved:
- reserve a lower explanatory lane beneath the main scatter instead of pinning captions to the bottom screen edge
- keep side labels in a separate lane with explicit horizontal gap from the chart/scatter
- clamp explanatory text blocks to the frame and verify them with geometry checks before animating
- use fade-out/fade-in when replacing one explanatory caption with another substantially different sentence

Not approved:
- captions that sit so low they are vulnerable to frame-edge crowding
- caption transitions that morph one sentence into another and become visually mushy mid-animation
- adding more decorative annotation instead of first fixing layout hierarchy and spacing

Why:
- text overlap problems in Manim scenes often come from not reserving real space for explanation
- even when objects do not technically overlap, crowded lower thirds still feel editorially wrong

Reusable rule:
- for annotation-heavy 2D scenes, define the text lanes first, then place the scatter/chart inside the remaining space
- verify:
  1. top stack clearance
  2. chart-to-caption clearance
  3. side-label horizontal gap
  4. final lower-block fit inside frame

## Default Refinement Standard For Future Scenes

For any remaining scene file in this repo, the target standard should be:

- easy to read as code
- easy to follow as animation
- visually consistent with the stronger scenes
- faithful to the scene description
- honest about what is simplified
- efficient enough to iterate on quickly
