import sys
import logging
from google.protobuf.json_format import Parse, MessageToJson
from utils.logger import Logger


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