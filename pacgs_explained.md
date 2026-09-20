# How the new 3D pipeline works, in plain English

**What we make:** you give the computer one ordinary photo of a person, standing, facing the camera.
It gives back a 3D model of that person that you can spin around and look at from any side.

**The 3D model** is a "Gaussian splat": instead of a solid surface like a video-game character, the
person is built from a few hundred thousand tiny coloured blobs floating in space. From any viewpoint
the blobs blend into a photograph-like image. You can open one in a browser at
[superspl.at/editor](https://superspl.at/editor) and drag it around.

**The hard part:** a photo shows the front only. The back of her head, the back of her jeans, the soles
of her shoes: none of that is in the photo, so the computer has to invent it. Everything below is about
inventing the missing sides *in a way that agrees with itself*.

---

## Why the earlier version struggled

The previous pipeline (call it the old way) did this:

1. Cut the person out of the photo.
2. Ask an image AI to redraw her from other angles: "same woman, seen from the left", "from behind",
   and so on. About a dozen separate pictures.
3. Work out, for each redrawn picture, where the camera must have been. It did that by finding her
   joints (shoulders, hips, knees) and solving backwards.
4. Build the 3D model from all those pictures.

Two things go wrong.

**The pictures disagree with each other.** Each one is drawn independently, so the hair falls a little
differently, the jeans pocket moves, the face changes slightly. When the computer tries to build one 3D
person out of a dozen slightly different people, it gets a blurry compromise, and the face in
particular comes out blotchy and doubled when you look from anywhere but straight on.

**Working out the camera afterwards is imprecise.** The estimate was typically 5–12 cm off. If you
think the camera was 10 cm to the left of where it really was, every detail in that picture is pasted
onto the 3D model in slightly the wrong place. Do that a dozen times and everything smears.

---

## The new way, step by step

The idea: **decide the camera path yourself, and make the AI follow it.** Then you never have to work
out where the camera was, because you chose it.

Think of it like this. Instead of asking an artist to draw the person from twelve angles and then
guessing where the artist was standing, we build a rough mannequin, walk a camera around it on a track
we laid ourselves, and ask the artist to paint over each frame, keeping the mannequin's exact shape and
position.

### Step 1: Cut her out, and fit a rough body

We already had this. The person is separated from the background, and a rough 3D body (the sort of
shape a mannequin has) is fitted to her pose. This also tells us where the camera was for the
original photo, in real-world terms: about 2.8 metres away, roughly at hip height, tilted up 5 degrees.

### Step 2: Use the *previous* 3D model as the stand-in

Here's the part that makes the whole thing work. We already had a 3D model of her from the old
pipeline. It's imperfect: the face breaks up at angles and there are stray blobs. But its overall
**shape** is fine, and, unlike the plain mannequin, it has her hair volume, her clothes and her heels.

So we use the old model as a stand-in, and the new pipeline's only job is to make it *look* right.

### Step 3: Film a turntable of the stand-in, as a depth video

We set up a camera path: a full circle around her, 81 frames, starting *exactly* where the real photo
was taken. Then we render the stand-in from each of those 81 positions, not in colour, but as a **depth
video**: a black-and-white clip where bright means near the camera and dark means far.

A depth video is the instruction sheet. It says, for every frame: here is the outline of the person,
here is how her body turns, here's where her hair sticks out, here's the shape of her arms.

Rendering all 81 frames takes about two seconds.

### Step 4: Let a video AI repaint it as the real person

We hand three things to a video AI (Wan 2.2 Fun-Control, running on your own machine):

- the depth video, which it must follow exactly
- **your photo as the first frame**, so the video literally starts on the real her
- your photo again as a reference for who she is

It produces 81 photo-realistic frames of *her*, turning all the way round, following the depth video
frame by frame. This takes about 8 minutes, most of it loading the AI model from the hard disk.

Two properties matter enormously here:

- A video AI generates all 81 frames *together*, so it keeps the same person, the same top, the same
  hair, the same shoes throughout. That's the opposite of a dozen separately drawn pictures.
- Because it follows the depth video, the person in frame 40 is exactly where our camera path says she
  should be. **We know the camera for every frame, exactly, because we chose it.**

### Step 5: Build the 3D model

We cut the person out of all 81 frames, and hand the trainer the frames plus their known cameras, plus
the original photo at full resolution. The trainer spends about 45 minutes arranging several hundred
thousand blobs until they reproduce every frame from its own camera.

---

## Why it worked

**We measured it three ways.**

**1. The video agrees with itself.** We hid 10 of the 81 frames from the trainer, built the model from
the other 71, and then asked it to render the 10 hidden viewpoints. The renders matched the hidden
frames at 30.4 dB. (Decibels here are a similarity score: 20 is a rough match, 30 is close, 40 is
nearly identical.)

For comparison, we ran the same test on a *perfect* video: we took the old 3D model and simply rendered
81 frames of it, which are flawless by construction, and trained on those. That scored 29.7 dB. **The
AI-generated video is as self-consistent as a video rendered from an actual 3D model.** That's the
single most surprising result of the day.

**2. The cameras really are exact.** We gave the trainer permission to nudge each frame's camera if it
thought that would fit better. It moved them by a quarter of a degree on average, and the result got
distinctly worse: 30.4 dB fell to 23.6 dB, and stray floating blobs multiplied eightfold. When
correcting something makes it worse, the thing was already right.

**3. It fixes the old pipeline's worst flaw.** Looking at her face from 30 degrees to the side, the old
models show patchy, doubled skin, like a badly stitched collage. The new one shows a clean, coherent
face. Stray blobs floating around her also dropped to a tenth: 0.014% of the image versus 0.145%.

**The short version of why:** every frame is consistent with every other frame, and every camera is
known rather than guessed. The trainer no longer has to average away disagreements.

---

## What's still wrong

**The face is still softer than the video it was trained on.** Both original causes are addressed (the
head close-up clip supplies the pixels, and the pasted-on layer is gone), but the model reproduces
about two thirds of the fine detail present in its own training frames. The reconstruction, not the
video, is now the limit for the face.

**Looking down from above used to show streaks** on her arms and the top of her head, because every
camera path was a flat circle at the photo's height and nothing in the video ever looked down at her.
Fix 4 below deals with this; what is left up there is softness, not smearing.

**The sides and back are softer than the front.** The photo is 1866×2799; the video frames are
480×832. The front is built from a sharp photo, everything else from small frames.

---

## Fixes we have already made

**1. A close-up pass on her head — done, and it worked.** The same trick, but the camera zooms in on
her head as it goes round, so her face fills the frame instead of being 40 pixels wide. The first frame
is a crop of your real photo, so it starts from genuine detail.

The result: measured on 15 camera angles around her face, the *worst* angle improved by a third
(a face-recognition score of 0.30 to 0.41), and the average became the best of any version we have
built, including the older pipeline's best. In plain terms: at 30 degrees off-centre you now see real
skin texture and individual strands of hair, instead of the old blotches or the slightly waxy look of
the first orbit-video model.

**2. A rising camera path — done.** A second clip where the camera climbs to 30 degrees above her as it
circles, so the top of her head and her shoulders are actually seen. The streaks on her arms and crown
are reduced, and the full-resolution front view improved too.

**3. Removing the pasted-on face — done, and this was the big one.** Looking closely, the face was
sharp from the front and speckled from every other angle, as if a dirty layer sat over a good face.
That is almost exactly what was happening. The pipeline used to paste a layer of flat blobs over the
face, one per pixel of your photo, and delete the real face underneath. Straight on, those blobs line
up into your photo perfectly. Turn 30 degrees and they tile apart, gaps open between them, and there is
nothing behind the gaps.

We now skip that layer entirely, stop forcing the face onto the rough body shape, and give the close-up
frames a proper share of the training time. Measured from ten camera angles around the head, the
"wrong detail" score fell from 1.09 to 0.83 (below 1.0 means the errors are now smaller than the real
detail), and agreement with the close-up frames rose from 25.4 to 32.2 dB. The front view did not pay
for it: it is slightly better than before.

**4. Close-ups from above and below, and over the top of her head — done.** The head close-up was a
single ring at the photo's height, so the face was only ever seen from one elevation up close. Tilt the
camera and it got worse fast. There are now four head rings instead of one: level with her, tilted up,
tilted down, steeply above, and one that looks straight down on the crown of her head.

Measured at those cameras, agreement at the steep angles went from 20.6 to 28.3 dB (about 35 degrees
above) and from 17.4 to 25.8 dB (over the crown). In plain terms: the old model turned her raised arms
into melted, shredded shapes when you looked down at her; the new one keeps arms, hands and hair.

**A lesson worth repeating, because it cost us a round.** For the over-the-head clip we first used the
rough *body shape* from step 1 as the stand-in instead of the previous 3D model. That body shape is
bare: no hair, no clothes. The video AI follows the shape it is given, so it obediently scraped her
hair into a bun and turned her strapped vest into a strapless one, and training on those frames dragged
every other view towards them. Regenerating from the proper stand-in fixed it. The pipeline now checks
each clip against the person's own outline and throws away any clip that disagrees by more than 10 per
cent, which is exactly what that failure looked like.

All of these are part of the normal pipeline: double-clicking `run_image_to_splat.bat` films seven
clips (body orbit, four head close-ups, a rising body orbit) and trains on them together. That takes
about three hours; `--no_orbit` gives the old, faster pipeline in about 35 minutes.

## What we're trying next

**1. Bigger frames.** Generating at 720×1280 instead of 480×832, roughly 2.5 times slower, for sharper
sides and back. There's also a cheaper option: an AI video upscaler that enlarges the finished clip
while keeping the frames consistent with each other.

**2. A second round.** The new model is better than the one we used as a stand-in. So use the new one
as the stand-in and go round again: better instruction sheet, better video, better model.

**3. More detail per face pixel.** The pasted-on face layer is gone (fix 3), and the model now
reproduces about two thirds of the detail in its own close-up frames. Two ways to raise that: upscale
the close-up frames with an AI image model before training on them, and add a pass that moves the
camera in close over the body rather than only the head.

**4. Test it on other photos.** Everything so far is one subject. Different hair, looser clothes and
different poses are where a method like this usually shows its limits.

---

## A few terms, briefly

| Term | What it means here |
|---|---|
| Gaussian splat | The 3D model: hundreds of thousands of tiny coloured blobs that look like a photo from any angle |
| Depth video | A black-and-white clip where brightness means distance. Used as the instruction sheet for the AI |
| Camera path | The route we move the virtual camera along: here, a circle around the person |
| Control (video) | Anything the AI is required to follow exactly, in our case the depth video |
| Frame | A single still picture out of the 81 in a clip |
| Held-out frames | Frames deliberately hidden from the trainer, kept to test the model honestly |
| dB | A similarity score between two images: 20 rough, 30 close, 40 nearly identical |

---

## Where things are

| | |
|---|---|
| The best model (all seven clips, face and crown fixed) | `output\pacgs\test-image\r1\splats\f12_dome40_hi2\character.ply` |
| Spin it around | `output\pacgs\test-image\r1\eval\f12_dome40_hi2\turntable.mp4` |
| The face from fifty angles, next to the frames it should match | `output\pacgs\test-image\r1\report\facecheck\facecheck.jpg` |
| The first orbit-video model (body clip only) | `output\pacgs\test-image\r1\splats\g1_splat360_l4_A\character.ply` |
| Every version side by side | `output\pacgs\test-image\r1\report\compare.png` |
| The generated video frames | `output\pacgs\test-image\r1\gen\<clip>\arc0\` |
| The technical write-up | [pacgs_results_sprint1.md](next-stage-research/docs/pacgs_results_sprint1.md) (first round) and [pacgs_face_fix.md](next-stage-research/docs/pacgs_face_fix.md) (the face and crown rounds, with figures) |
| What we plan to try next, in order | [pacgs_quality_next.md](next-stage-research/docs/pacgs_quality_next.md) |
