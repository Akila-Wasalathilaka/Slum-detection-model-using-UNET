# Colab: Generate 15–20 Overlays in Batch

Use the notebook `colab_generate_overlays.ipynb` to run inference over a random subset of your test images and produce overlays and probability maps.

Steps
- Open Google Colab.
- File > Open notebook > GitHub tab > paste repo URL or upload the notebook `colab_generate_overlays.ipynb`.
- In the Setup cell, it clones the repo and installs requirements.
- Set the checkpoint path (default `best_global_model.pth`) and ensure your images are under `data/test/images`.
- Run the Batch inference cell: it samples up to 20 images and writes results into `colab_outputs/`.
- Optionally run the final cell to download a zip of the results.

Outputs
- colab_outputs/*_overlay.jpg
- colab_outputs/*_prob.jpg
- colab_outputs/batch_results.json (includes threshold and mean confidence per image)

Tune
- Adjust N_SAMPLES between 15–20 as needed.
- Toggle `use_tta`/`use_tent` by editing the call in the batch cell if you need faster runs.
