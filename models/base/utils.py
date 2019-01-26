from json import dump
from os import makedirs
from os.path import join, exists
from uuid import uuid4
from google.protobuf.json_format import MessageToDict


def gen_model_name(prefix=''):
    return '{}{:X}'.format(prefix, uuid4().int >> 104)


def get_logs_dir(options, model_name=None):
    if not exists(options.storage.logs):
        makedirs(options.storage.logs)

    model_name = model_name or gen_model_name()
    logs_dir = join(options.storage.logs, model_name)
    makedirs(logs_dir)
    options_dict = MessageToDict(
        message=options, including_default_value_fields=True, preserving_proto_field_name=True)

    with open(join(logs_dir, 'options.json'), 'w') as f:
        dump(options_dict, f, sort_keys=True, indent=2)

    return logs_dir
