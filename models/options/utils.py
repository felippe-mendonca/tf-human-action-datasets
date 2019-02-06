import sys
import logging
from google.protobuf.json_format import Parse, MessageToJson, MessageToDict
from utils.logger import Logger
from json import dumps


def load_options(filename, schema):
    log = Logger(name='LoadOptions')

    try:
        with open(filename, 'r') as f:
            try:
                options = Parse(f.read(), schema())
                log.info(
                    "\n{}: \n{}", schema.DESCRIPTOR.full_name,
                    MessageToJson(
                        options,
                        including_default_value_fields=True,
                        preserving_proto_field_name=True))
                return options
            except Exception as ex:
                log.critical("Unable to load options from '{}'. \n{}", filename, ex)
            except:
                log.critical("Unable to load options from '{}'", filename)
    except Exception as ex:
        log.critical("Unable to open file '{}'", filename)


def make_description(options, remove_fields=None, indent=2):
    op_dict = MessageToDict(
        message=options, including_default_value_fields=True, preserving_proto_field_name=True)

    hidden_fields = ['telegram', 'storage', 'estimator']
    if remove_fields is not None:
        hidden_fields += remove_fields
    for field in hidden_fields:
        if field in op_dict:
            del op_dict[field]
    return dumps(op_dict, indent=indent, sort_keys=True)