# learn_rl

## Installation Guide

Clone the repository:

```sh
git clone https://github.com/tek4052/learn_rl.git
```

If not yet, install module `venv`:

```sh
sudo apt-get install python3-venv
```

Create a virtual environment:

```sh
cd learn_rl/
python3 -m venv venv
source venv/bin/activate
```

Install `pip-tools`:

```sh
pip install pip-tools
```

Compile the requirements:

```sh
pip-compile ./requirements.in
```

Install the requirements:

```sh
pip install -r requirements.txt
```
