Put your photos here (any file name).

A good input photo:
- one person, front-facing, full body from head to feet, with a little margin around them
- 1500 px tall or more; plain, uncluttered backgrounds cut out best
- png, jpg or webp

Then either drag the photo onto `run_image_to_splat.bat`, or run:

    python3 image_to_splat.py --input input\your_photo.png

The result is written to `output\<photo name>\v2\final\character.ply`, unless you pass `--output_dir`.
The 121-frame `*_static_4danyone.mp4` files belong to the old 4DAnyone path; v2 does not need them.
