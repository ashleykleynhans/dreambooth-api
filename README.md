# A collection of Python scripts for calling the REST API for the Dreambooth extension to the Automatic1111 Stable Diffusion WebUI

## Overview

This is a collection of Python scripts for calling the REST
API of the [Dreambooth extension](
https://github.com/d8ahazard/sd_dreambooth_extension) for the
[AUTOMATIC1111 Stable Diffusion Web UI](
https://github.com/AUTOMATIC1111/stable-diffusion-webui).

After extensive testing, I have determined that the
[v1.2.1](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.2.1)
release of the Web UI, and the [1.0.14 (unrelessed)](
https://github.com/d8ahazard/sd_dreambooth_extension/releases/tag/1.0.14)
version of the Dreambooth extension produce the best results.

I have also discovered that it is better to disable `xformers`
for training and rather set memory attention to `default`, and
that Torch version `1.13.1` works better than Torch version 2.

You can downgrade Torch as follows

```bash
cd stable-diffusion-webui
source venv/bin/activate
pip3 install torch==1.13.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip uninstall xformers
pip install https://huggingface.co/MonsterMMORPG/SECourses/resolve/main/xformers-0.0.19-cp310-cp310-manylinux2014_x86_64.whl
```

You should ensure that the versions of the Python dependencies
in the `requirements.txt` file for the Dreambooth extension
match the versions in the `requirments.txt` and
`requirements_versions.txt` files of the Automatic1111 Web
UI, and update the ones that don't match in the Automatic1111
Web UI, otherwise the Web UI will revert the dependencies
each time its started and will result in a slow start-up time.

It is recommended to use a GPU instance with at least 16GB
of VRAM, but preferably more to prevent GPU OOM errors,
and you will need at least 32GB of memory for Dreambooth
training, otherwise the Web UI will be OOM killed by the
system kernel.

## Step 1 - Clone the repository

```bash
git clone https://github.com/ashleykleynhans/dreambooth-api.git
```

## Step 2 - Configure your Python Virtual Environment

Obviously this assumes that you already have Python
installed on your system.

Change directory to the repository that you cloned above:

```bash
cd dreambooth-api
```

Create a new Python virtual environment:

```bash
python3 -m venv venv
```

Activate the Python virtual environment:

```bash
source venv/bin/activate
```

Install the required Python dependencies:

```bash
pip3 install -r requirements.txt
```

## Step 3 - Update the configuration file

Edit `config.yml` and update the appropriate
configuration settings.

## Step 4 - Create a new Dreambooth model to train

You need to start of by creating a new Dreambooth model
to train.

You can do this by running the script:

```bash
python3 create_model.py
```

## Step 5 - Configure your newly created Dreambooth model

Once you have created your new Dreambooth model for
training, you need to configure it.

You can do this by running the script:

```bash
python3 set_model_config.py
```

## Step 6 - Start training your Dreambooth model

Once you have created your new Dreambooth model and
configured it, you can proceed to train it.

You can do this by running the script:

```bash
python3 start_training.py
```

Training can be cancelled at any time, by running
the `cancel_training.py` script (see Step 8 below).

## Step 7 - Check Status

Once you have started training your new Dreambooth
model, you can check the progress and get status
images.

### Step 7 (a) - Check Progress

You can check the progress of your training by
running the script:

```bash
python3 get_status.py
```

### Step 7 (b) - Get Status Images

You can do this by running the script:

```bash
python3 get_status_images.py
```

## Step 8 - Cancel Training (Optional)

If you find that your sample images are over-trained,
or if you want to cancel training for any other
reason, you can do so by running the script:

```bash
python3 cancel_training.py
```
