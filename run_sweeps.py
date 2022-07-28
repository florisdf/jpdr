import argparse
import os
from pathlib import Path
import requests
import sys
import json
import yaml
from yaml import SafeLoader

from dotenv import load_dotenv
import wandb


load_dotenv()

WEBHOOK_URL = (
    "https://hooks.slack.com/services/"
    f"{os.getenv('SLACK_WEBHOOK_KEY')}"
)


def check_conf(conf):
    conf_path = Path(conf)
    if not conf_path.exists():
        raise ValueError(
            f'File "{conf}" not found'
        )
    return conf_path


def notify_slack(title, message):
    slack_data = {
        "username": "Wandb sweep runner",
        "icon_emoji": ":broom:",
        "attachments": [
            {
                "fields": [
                    {
                        "title": title,
                        "value": message,
                        "short": "false",
                    }
                ]
            }
        ]
    }
    byte_length = str(sys.getsizeof(slack_data))
    headers = {'Content-Type': "application/json",
               'Content-Length': byte_length}
    response = requests.post(WEBHOOK_URL, data=json.dumps(slack_data),
                             headers=headers)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'sweep_confs', nargs='+', help='The sweeps configs',
        type=check_conf
    )

    args = parser.parse_args()

    for conf_path in args.sweep_confs:
        conf_dict = yaml.load(conf_path.open(), SafeLoader)
        project = conf_dict['project']
        entity = conf_dict['entity']
        sweep_id = wandb.sweep(
            conf_dict,
            project=project,
            entity=entity
        )
        sweep_url = f"https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}"
        notify_slack(
            f"Started sweep {sweep_id}",
            f"This is the sweep from {conf_path}. "
            f"View the sweep at {sweep_url}"
        )
        wandb.agent(
            sweep_id,
            project=project,
            entity=entity
        )
        notify_slack(
            f"Finished sweep {sweep_id}",
            f"This is the sweep from {conf_path}. "
            f"View the sweep at {sweep_url}"
        )
