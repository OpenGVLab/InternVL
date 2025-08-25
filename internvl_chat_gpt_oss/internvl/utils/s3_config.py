# -*- coding: utf-8 -*-

import configparser
import logging

import internvl.utils.s3_exception as exception

DEFAULT_SECTION_NAME = 'DEFAULT'


# 如果配置文件中有 [DEFAULT]，对应内容将覆盖此处
CONFIG_DEFAULT = {
    'endpoint_url': '%(host_base)s',
    'file_log_level': 'DEBUG',
    'file_log_max_bytes': 1024 * 1024 * 1024,  # 1GB
    'file_log_backup_count': 1,
    'console_log_level': 'WARNING',
    'count_disp': '5000',
    'enable_mem_trace': 'False',
    'get_retry_max': '10',
    's3_cpp_log_level': 'off',
    'cpp_multipart_threads': 10,
    'cpp_multipart_chunk_size': 5*1024*1024,
    'address_style': "non-virtual",
}

_UNSET = object()


def _value_to_str(d):
    if isinstance(d, (int, bool)):
        return str(d)
    if isinstance(d, (dict,)):
        return {
            k: _value_to_str(v) for k, v in d.items()
        }
    return d


class GetterMixin(object):

    _boolean_states = {'1': True, 'yes': True, 'true': True, 'on': True,
                       '0': False, 'no': False, 'false': False, 'off': False}

    def get(self, key, default=_UNSET):
        try:
            return self[key]
        except exception.ConfigItemNotFoundError:
            if default is _UNSET:
                raise
            else:
                return default

    def has_option(self, key):
        try:
            self[key]
            return True
        except exception.ConfigItemNotFoundError:
            return False

    def get_boolean(self, key, default=_UNSET):
        v = str(self.get(key, default)).lower()

        if v not in self._boolean_states:
            raise exception.ConfigKeyTypeError('Not a boolean: ' + key)
        return self._boolean_states[v]

    def get_int(self, key, default=_UNSET):
        try:
            return int(self.get(key, default))
        except ValueError:
            raise exception.ConfigKeyTypeError('Not a integer: ' + key)

    def get_log_level(self, key, default=_UNSET):
        v = str(self.get(key, default)).upper()
        if v not in logging._nameToLevel:
            raise exception.ConfigKeyTypeError('Not a log level: ' + key)
        return logging._nameToLevel[v]


class _my_dict(configparser._default_dict):
    pass


class Config(GetterMixin):
    def __init__(self, conf_path, *args, **kwargs):
        parser = configparser.ConfigParser(CONFIG_DEFAULT)
        r = parser.read(conf_path, encoding='utf-8')
        if len(r) == 0:
            raise exception.ConfigFileNotFoundError(conf_path)
        if len(parser.sections()) == 0:
            raise exception.ConfigSectionNotFoundError()

        defaults = parser._defaults
        all_sections = parser._sections.items()
        deleteList = []
        for section, options in all_sections:
            if section.lower() != "default":
                continue
            for name, val in options.items():
                defaults[name] = val
            deleteList.append(section)
        for deleteSection in deleteList:
            parser.remove_section(deleteSection)

        self._parser = parser
        self._default = parser.items(DEFAULT_SECTION_NAME, raw=True)

    def __getitem__(self, key):
        try:
            return Section(self._parser[key])
        except KeyError as err:
            raise exception.ConfigSectionNotFoundError(*err.args)

    def update(self, other: dict):
        for k, v in other.items():
            self._parser[k].update(_value_to_str(v))

    def default(self):
        return Section(dict(self._default))

    def items(self):
        sections = self._parser.sections()
        if len(sections) == 0:
            raise exception.ConfigSectionNotFoundError()
        return [(section, self[section]) for section in sections]


class Section(GetterMixin):

    def __init__(self, conf: dict):
        # 注意 conf 中 value 取值类型均为 str
        self._conf = conf

    def __getitem__(self, key):
        try:
            return self._conf[key]
        except KeyError as err:
            raise exception.ConfigKeyNotFoundError(*err.args)

    def update(self, other):
        self._conf.update(_value_to_str(other))
