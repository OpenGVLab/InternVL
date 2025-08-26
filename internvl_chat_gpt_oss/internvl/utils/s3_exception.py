# -*- coding: utf-8 -*-


class Error(Exception):
    '''Base class for exceptions in petrel_oss module.'''

    def __str__(self):
        cls_name = type(self).__name__
        msg = super(Error, self).__str__()
        return '{}({})'.format(cls_name, msg)


class RetriableError(Error):
    pass


# Config Error


class ConfigError(Error):
    pass


class InvalidConfigError(ConfigError):
    pass


class ConfigFileNotFoundError(ConfigError):
    pass


class ConfigItemNotFoundError(ConfigError):
    pass


class ConfigKeyNotFoundError(ConfigItemNotFoundError):
    pass


class ConfigSectionNotFoundError(ConfigItemNotFoundError):
    pass


class ConfigKeyTypeError(ConfigError):
    pass


class ConfigKeyValueError(ConfigError):
    pass


class UnSupprotAddressStyle(ConfigError):
    pass


# Client Error


class ClientError(Error):
    pass


class ContentTypeError(ClientError):
    pass


class S3ClientError(ClientError):
    pass


class InvalidAccessKeyError(S3ClientError):
    pass


class SignatureNotMatchError(S3ClientError):
    pass


class NetworkConnectionError(S3ClientError):
    pass


class ResourceNotFoundError(S3ClientError):
    pass


class AccessDeniedError(ClientError):
    pass


class RangeError(ClientError):
    pass


class MultipartError(ClientError):
    pass


class ObjectNotFoundError(ClientError):
    pass


class S3ObjectNotFoundError(ObjectNotFoundError):
    pass


class NoSuchBucketError(S3ObjectNotFoundError):
    pass


class NoSuchKeyError(S3ObjectNotFoundError):
    pass


# Cache Error


class CacheError(ClientError):
    pass


class McClientError(CacheError):
    pass


class McObjectNotFoundError(ObjectNotFoundError, McClientError):
    pass


class McTimeoutOccur(McClientError, RetriableError):
    pass


class McConnFailed(McClientError, RetriableError):
    pass


class McServerFailed(McClientError, RetriableError):
    pass


class McServerDisable(McClientError):
    pass


class McServerDead(McClientError):
    pass


class McBadKeyProvided(McClientError):
    pass


class McKeySizeExceed(McClientError):
    pass


class McObjectSizeExceed(McClientError):
    pass


# URI Error


class InvalidUriError(Error):
    pass


class InvalidS3UriError(InvalidUriError):
    pass


class InvalidBucketUriError(InvalidS3UriError):
    pass


class InvalidDfsUriError(InvalidUriError):
    pass


class InvalidMcUriError(InvalidUriError):
    pass


class InvalidClusterNameError(InvalidUriError):
    pass


class NoDefaultClusterNameError(InvalidUriError):
    pass
