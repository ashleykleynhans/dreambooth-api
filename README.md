# Sample REST API scripts for the Dreambooth Extension to the Stable Diffusion WebUI

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

## Step 3 - Create a new Dreambooth model to train

You need to start of by creating a new Dreambooth model
to train.

You can do this by editing the `create_model.py` script,
updating `URL` to the URL of your Stable Diffusion WebUI
instance, updating `NEW_MODEL_NAME` to the name of the new
Dreambooth model that you would like to create, updating
`SRC_MODEL` to the name of the model that you would like
to base your new Dreambooth model off of, and then running
the script:

```bash
python3 create_model.py
```

## Step 4 - Configure your newly created Dreambooth model

Once you have created your new Dreambooth model for
training, you need to configure it.

You can do this by editing the `model_config.py` script,
updating `URL` to the URL of your Stable Diffusion WebUI
instance, updating the rest of the variables to your
requirements for training purposes, and then running the
script:

```bash
python3 model_config.py
```

## Step 5 - Start training your Dreambooth model

Once you have created your new Dreambooth model and
configured it, you can proceed to train it.

You can do this by editing the `start_training.py` script,
updating `URL` to the URL of your Stable Diffusion WebUI
instance, updating `MODEL_NAME` to your newly created
Dreambooth model, and then running the script:

```bash
python3 start_training.py
```

Training can be cancelled at any time, by running
the `cancel_training.py` script (see Step 7 below).

## Step 6 - Check Status

Once you have started training your new Dreambooth
model, you can check the progress and get status
images.

### Step 6 (a) - Check Progress

__TODO__: There is currently an issue with the API,
see the [Github issue](https://github.com/d8ahazard/sd_dreambooth_extension/issues/1228).

### Step 6 (b) - Get Status Images

You can do this by editing the `get_status_images.py`
script, updating `URL` to the URL of your Stable Diffusion
WebUI instance, and then running the script:

```bash
python3 get_status_images.py
```

## Step 7 - Cancel Training (Optional)

If you find that your sample images are over-trained,
or if you want to cancel training for any other
reason, you can do so by editing the
`cancel_training.py` script, updating `URL` to the
URL of your Stable Diffusion WebUI instance, and
then running the script:

```bash
python3 cancel_training.py
```
