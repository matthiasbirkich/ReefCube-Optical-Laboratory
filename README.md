# ReefCube-Optical-Laboratory
This is a tool for calculating data from the underwater spectral light sensor AS7343

Google Colab and private GitHub repository guide

Recommended repository structure

```text
ReefCube-Optical-Laboratory/
├── README.md
├── requirements.txt
├── reefcube/
│   ├── __init__.py
│   ├── analysis.py
│   ├── calibration.py
│   ├── config.py
│   ├── constants.py
│   ├── deployment.py
│   ├── measurement.py
│   ├── optics.py
│   ├── ppfd.py
│   ├── sensors.py
│   ├── spectroscopy.py
│   ├── storage.py
│   ├── visualization.py
│   └── wavelength.py
└── notebooks/
    └── ReefCube_Optical_Laboratory_complete_demo.ipynb
```

Upload to GitHub

Do not upload the project as one ZIP if you want GitHub and Colab to use the files directly.

1. Create the new private repository.
2. Open the repository in Safari or on a computer.
3. Select Add file → Upload files.
4. Upload README.md, requirements.txt, the reefcube folder, and the notebooks folder.
5. Commit the uploaded files to the main branch.

Create a GitHub token for Colab

1. In GitHub, open Settings.
2. Open Developer settings.
3. Choose Personal access tokens → Fine-grained tokens.
4. Create a token restricted to ReefCube-Optical-Laboratory.
5. Grant Contents: Read-only permission.
6. Copy the token.

Store the token in Colab

1. Open the notebook in Google Colab.
2. In the left sidebar, select the key symbol Secrets.
3. Add a secret named exactly GITHUB_TOKEN.
4. Paste the token.
5. Enable notebook access.

Never paste the token into a notebook code cell and never commit it to GitHub.

Run the notebook

Use Runtime → Run all.

Commands beginning with ! are shell commands executed by Colab.
Normal Python code does not start with !.
The %cd syntax changes Colab’s working directory, although the supplied notebook handles this automatically with Python.

Open the notebook from GitHub

In Colab:

1. Select File → Open notebook.
2. Select the GitHub tab.
3. Authorize GitHub access when requested.
4. Enter matthiasbirkich/ReefCube-Optical-Laboratory.
5. Open notebooks/ReefCube_Optical_Laboratory_complete_demo.ipynb.

For a private repository, GitHub authorization and the Colab GITHUB_TOKEN secret may both be needed: authorization lets Colab list/open the notebook; the secret lets the running notebook clone the complete repository.
