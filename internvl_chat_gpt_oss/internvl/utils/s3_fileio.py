
# Copied from https://github.com/open-mmlab/mmengine/pull/984
# Copyright (c) OpenMMLab. All rights reserved.
import io
import os
import re
import tempfile
import threading
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Iterator, Optional, Tuple, Union
from os.path import expanduser, abspath
import functools
import internvl.utils.s3_exception as exception
from internvl.utils.s3_config import Config

LOG = logging.getLogger(__name__)

_S3_URI_PATTERN = re.compile(
    r'^(?:(?P<cluster>[^:]+):)?(s3://)?(?P<bucket>[^/]+)/?(?P<key>(?:.+?)/?$)?', re.I)

from mmengine.fileio.backends.base import BaseStorageBackend
thread_local_client = threading.local()

class S3Backend(BaseStorageBackend):
    """S3 storage bachend.

    S3Backend supports reading and writing data to aws
    It relies on awscli and boto3, you must install and run ``aws configure``
    in advance to use it.

    Args:
        path_mapping (dict, optional): Path mapping dict from local path to
            Petrel path. When ``path_mapping={'src': 'dst'}``, ``src`` in
            ``filepath`` will be replaced by ``dst``. Default: None.

    Examples:
        >>> filepath = 's3://bucket/obj'
        >>> backend = S3Backend()
        >>> backend.get(filepath1)  # get data from s3
        b'hello world'
    """

    def __init__(self,
                 end_point_url: Optional[str] = None,
                 access_key_id: Optional[str] = None,
                 secret_access_key: Optional[str] = None,
                 path_mapping: Optional[dict] = None):
        try:
            import boto3
            from botocore.exceptions import ClientError
            from botocore.config import Config
        except ImportError:
            raise ImportError('Please install boto3 to enable '
                              'S3Backend.')
        self.config = Config(   
            read_timeout=5,
            connect_timeout=2
        )
        if access_key_id and secret_access_key:
            self._client = boto3.client(
                's3',
                endpoint_url=end_point_url,
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                config=self.config)
        else:
            self._client = boto3.client('s3')
        assert isinstance(path_mapping, dict) or path_mapping is None
        self.path_mapping = path_mapping
        # Use to parse bucket and obj_name
        self.parse_bucket = re.compile('s3://(.+?)/(.+)')
        self.check_exception = ClientError

    def _map_path(self, filepath: Union[str, Path]) -> str:
        """Map ``filepath`` to a string path whose prefix will be replaced by
        :attr:`self.path_mapping`.

        Args:
            filepath (str or Path): Path to be mapped.
        """
        filepath = str(filepath)
        if self.path_mapping is not None:
            for k, v in self.path_mapping.items():
                filepath = filepath.replace(k, v, 1)
        return filepath

    def _format_path(self, filepath: str) -> str:
        """Convert a ``filepath`` to standard format of aws.

        If the ``filepath`` is concatenated by ``os.path.join``, in a Windows
        environment, the ``filepath`` will be the format of
        's3://bucket_name\\image.jpg'. By invoking :meth:`_format_path`, the
        above ``filepath`` will be converted to 's3://bucket_name/image.jpg'.

        Args:
            filepath (str): Path to be formatted.
        """
        return re.sub(r'\\+', '/', filepath)

    def _parse_path(self, filepath: Union[str, Path]) -> Tuple[str, str]:
        """Parse bucket and object name from a given ``filepath``.
        Args:
            filepath (str or Path): Path to read data.

        Returns:
            bucket (str): Bucket name of aws s3.
            obj_name (str): Object relative path to bucket.
        """
        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        parse_res = self.parse_bucket.findall(filepath)
        if not parse_res:
            raise ValueError(
                f"The input path '{filepath}' format is incorrect."
                'Correct example: s3://bucket/file')
        bucket, obj_name = parse_res[0]
        return bucket, obj_name

    def _check_bucket(self, bucket: str) -> bool:
        """Check if bucket exists.

        Args:
            bucket (str): Bucket name
        Returns:
            bool: True if bucket is existing.
        """
        try:
            self._client.head_bucket(Bucket=bucket)
            return True
        except self.check_exception:
            LOG.warning(f'Bucket {bucket} does not exist or is not accessible.')
            return False

    def _check_object(self, bucket: str, obj_name: str) -> bool:
        """Check if object exists.

        Args:
            bucket (str): Bucket name
            obj_name (str): Object name
        Returns:
            bool: True if object is existing.
        """
        try:
            self._client.head_object(Bucket=bucket, Key=obj_name)
            return True
        except self.check_exception:
            return False

    def get(self, filepath: str) -> bytes:
        """Read bytes from a given ``filepath``.

        Args:
            filepath (str): Path to read data.

        Returns:
            bytes: Expected bytes object.

        Examples:
            >>> backend = S3Backend()
            >>> backend.get('s3://bucket/file')
            b'hello world'
        """
        bucket, obj_name = self._parse_path(filepath)
        self._check_bucket(bucket)
        return self._client.get_object(
            Bucket=bucket, Key=obj_name)['Body'].read()

    def get_text(self, filepath, encoding='utf-8') -> str:
        """Read text from a given ``filepath``.

        Args:
            filepath (str): Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Defaults to 'utf-8'.

        Returns:
            str: Expected text reading from ``filepath``.

        Examples:
            >>> backend = S3Backend()
            >>> backend.get_text('s3://bucket/file')
            'hello world'
        """
        return self.get(filepath).decode('utf-8')

    def put(self, obj: bytes, filepath: Union[str, Path]) -> None:
        """Save data to a given ``filepath``.

        Args:
            obj (bytes): Data to be saved.
            filepath (str or Path): Path to write data.
        """
        bucket, obj_name = self._parse_path(filepath)
        self._check_bucket(bucket)
        with io.BytesIO(obj) as buff:
            # Todo: add progressPercentage
            self._client.upload_fileobj(buff, bucket, obj_name)

    def put_text(self,
                 obj: str,
                 filepath: Union[str, Path],
                 encoding: str = 'utf-8') -> None:
        """Save data to a given ``filepath``.

        Args:
            obj (str): Data to be written.
            filepath (str or Path): Path to write data.
            encoding (str): The encoding format used to encode the ``obj``.
                Default: 'utf-8'.
        """
        self.put(bytes(obj, encoding=encoding), filepath)

    def remove(self, filepath: Union[str, Path]) -> None:
        """Remove a file from aws s3.

        Args:
            filepath (str or Path): Path to be removed.
        """
        bucket, obj_name = self._parse_path(filepath)
        self._client.delete_object(Bucket=bucket, Key=obj_name)

    def exists(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path exists.

        Args:
            filepath (str or Path): Path to be checked whether exists.
        Returns:
            bool: Return ``True`` if ``filepath`` exists, ``False`` otherwise.
        """
        bucket, obj_name = self._parse_path(filepath)
        return self._check_object(bucket, obj_name)

    def isdir(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a directory.

        Args:
            filepath (str or Path): Path to be checked whether it is a
                directory.
        Returns:
            bool: Return ``True`` if ``filepath`` points to a directory,
            ``False`` otherwise.
        """
        bucket, obj_name = self._parse_path(filepath)
        if self._check_bucket(bucket) and (obj_name.endswith('/')
                                           or obj_name == ''):
            return True
        return False

    def isfile(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a file.

        Args:
            filepath (str or Path): Path to be checked whether it is a file.
        Returns:
            bool: Return ``True`` if ``filepath`` points to a file, ``False``
                otherwise.
        """
        bucket, obj_name = self._parse_path(filepath)
        if self._check_bucket(
                bucket) and not obj_name.endswith('/') and obj_name != '':
            return True
        return False

    @contextmanager
    def get_local_path(
            self, filepath: str) -> Generator[Union[str, Path], None, None]:
        """Download a file from ``filepath`` to a local temporary directory,
        and return the temporary path.

        ``get_local_path`` is decorated by :meth:`contxtlib.contextmanager`. It
        can be called with ``with`` statement, and when exists from the
        ``with`` statement, the temporary path will be released.

        Args:
            filepath (str): Download a file from ``filepath``.

        Yields:
            Iterable[str]: Only yield one temporary path.

        Examples:
            >>> backend = S3Backend()
            >>> # After existing from the ``with`` clause,
            >>> # the path will be removed
            >>> with backend.get_local_path('s3://path/of/file') as path:
            ...     # do something here
        """
        try:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.write(self.get(filepath))
            f.close()
            yield f.name
        finally:
            os.remove(f.name)

    def list(self,
             dir_path: Union[str, Path],
             list_dir: bool = True,
             list_file: bool = True,
             suffix: Optional[Union[str, Tuple[str]]] = None,
             recursive: bool = False) -> Iterator[str]:
        """Scan a directory to find the interested directories or files in
        arbitrary order.
        Note:
            s3 has no concept of directories but it simulates the directory
            hierarchy in the filesystem through public prefixes. In addition,
            if the returned path ends with '/', it means the path is a public
            prefix which is a logical directory.
        Note:
            :meth:`list_dir_or_file` returns the path relative to ``dir_path``.
        Args:
            dir_path (str | Path): Path of the directory.
            list_dir (bool): List the directories. Default: True.
            list_file (bool): List the path of files. Default: True.
            suffix (str or tuple[str], optional):  File suffix
                that we are interested in. Default: None.
            recursive (bool): If set to True, recursively scan the
                directory. Default: False.
            maxnum (int): The maximum number of list. Default: 1000.


        Yields:
            Iterable[str]: A relative path to ``dir_path``.
        """
        if list_dir and suffix is not None:
            raise TypeError(
                '`list_dir` should be False when `suffix` is not None')

        if (suffix is not None) and not isinstance(suffix, (str, tuple)):
            raise TypeError('`suffix` must be a string or tuple of strings')

        bucket, obj_name = self._parse_path(dir_path)
        dir_path = obj_name

        # AWS s3's simulated directory hierarchy assumes that directory paths
        # should end with `/` if it not equal to ''.
        if dir_path and not dir_path.endswith('/'):
            dir_path += '/'

        root = dir_path

        # Used to filter duplicate folder paths
        duplicate_paths = set()

        def _list_dir_or_file(dir_path,
                              list_dir,
                              list_file,
                              suffix,
                              recursive,
                              start_token=None):
            # boto3 list method, it return json data as follows:
            # {
            #     'ResponseMetadata': {..., 'HTTPStatusCode': 200, ...},
            #     ...,
            #     'Contents': [{'Key': 'path/object', ...}, ...],
            #     ...,
            #     'NextContinuationToken': '',
            #     ...
            # }
            paginator = self._client.get_paginator('list_objects_v2')
            pagination_config = {'MaxItems': 1000, 'PageSize': 1000}
            if start_token is not None:
                pagination_config.update({'StartingToken': start_token})
            response_iterator = paginator.paginate(
                Bucket=bucket,
                Prefix=dir_path,
                PaginationConfig=pagination_config)
            next_token = None
            for response in response_iterator:
                if 'NextContinuationToken' in response:
                    next_token = response['NextContinuationToken']
                if (response['ResponseMetadata']['HTTPStatusCode'] == 200
                        and 'Contents' in response):
                    for content in response['Contents']:
                        path = content['Key'][len(root):]
                        # AWS s3 has no concept of directories, it will list
                        # all path of object from bucket. Compute folder level
                        # to distinguish different folder.
                        sparse_path = [
                            item for item in path.replace(root, '').split('/')
                            if item
                        ]
                        level = len(sparse_path)
                        if level == 0:
                            continue
                        # If recursive is False, return only one level of
                        # directory.
                        if level > 1 and not recursive:
                            if list_dir and sparse_path[
                                    0] not in duplicate_paths:
                                yield sparse_path[0] + '/'
                                duplicate_paths.add(sparse_path[0])
                            continue
                        if list_dir:
                            # Resolve the existing folder path according to
                            # the path of the object. The folder path must end
                            # with '/'.
                            for lvl in range(level - 1):
                                rel_dir = '/'.join(sparse_path[:lvl + 1])
                                if rel_dir not in duplicate_paths:
                                    yield rel_dir + '/'
                                duplicate_paths.add(rel_dir)
                        if list_file and (suffix is None
                                          or path.endswith(suffix)):
                            yield path
            if next_token is not None:
                yield from _list_dir_or_file(
                    dir_path,
                    list_dir,
                    list_file,
                    suffix,
                    recursive,
                    start_token=next_token)

        return _list_dir_or_file(dir_path, list_dir, list_file, suffix,
                                 recursive)



class MixedClient(object):
    def __init__(self, conf_path, **kwargs):
        conf_path = abspath(expanduser(conf_path))
        config = Config(conf_path)
        self._default_config = config.default()

        # init_log(self._default_config)

        # LOG.debug('Init MixedClient, conf_path %s', conf_path)
        # Profiler.set_default_conf(self._default_config)

        self._ceph_dict = {
            cluster: S3Backend(end_point_url=conf['endpoint_url'], access_key_id=conf.get('access_key'),secret_access_key= conf.get('secret_key'))
            for cluster, conf in config.items() if cluster.lower() not in ('dfs', 'cache', 'mc')
        }
        # print(f'Ceph dict: {len(self._ceph_dict)}')
        self._default_cluster = self._default_config.get(
            'default_cluster', None)
        self._count_disp = self._default_config.get_int('count_disp')
        self._get_retry_max = self._default_config.get_int('get_retry_max')

    @staticmethod
    def parse_uri(uri, ceph_dict, default_cluster=None):
        m = _S3_URI_PATTERN.match(uri)
        if m:
            cluster, bucket, key = m.group(
                'cluster'), m.group('bucket'), m.group('key')
            cluster = cluster or default_cluster
            if not cluster:
                raise exception.NoDefaultClusterNameError(uri)
            '''
            try:
                new_key = ""
                for _, c in enumerate(key):
                    new_key.join(SPECIAL_CHAR_MAP[c] if SPECIAL_CHAR_MAP[c] else c)
                client = ceph_dict[cluster]
                return cluster, bucket, new_key, False
            except KeyError:
                raise InvalidClusterNameError(cluster)
            '''
            try:
                client = ceph_dict[cluster]
                return cluster, bucket, key, False
            except KeyError:
                raise exception.InvalidClusterNameError(cluster)
        else:
            raise exception.InvalidS3UriError(uri)


    def get_with_info(self, uri, **kwargs):     
        cluster, bucket, key, enable_cache = self.parse_uri(
            uri, self._ceph_dict, self._default_cluster)
        client = self._ceph_dict[cluster]
        # print(f'Get file from {cluster}:{bucket}/{key} with enable_cache={enable_cache}')
        filepath = f"s3://{bucket}/{key}"
        # print(f'Get file from {filepath}')
        info = { }
        return client.get(filepath), info

    def list(self, uri, **kwargs):
        cluster, bucket, key, _ = self.parse_uri(
            uri, self._ceph_dict, self._default_cluster)
        client = self._ceph_dict[cluster]
        dirpath = f"s3://{bucket}/{key}"
        return client.list(dir_path=dirpath, **kwargs)
    
class Client(object):

    def __init__(self, conf_path='petreloss.conf', *args, **kwargs):
        assert conf_path is not None, \
            'conf_path must be specified for Client initialization.'
        self._conf_path = conf_path
        self.kwargs = kwargs

        self._get_local_client()

    def _get_local_client(self): 
        current_pid = os.getpid()
        client, client_pid = getattr(
            thread_local_client,
            self._conf_path,
            (None, None)
        )
        if current_pid != client_pid:
            client = MixedClient(self._conf_path, **self.kwargs)
            setattr(
                thread_local_client,
                self._conf_path,
                (client, current_pid)
            )
        return client

    def get_with_info(self, uri, **kwargs):
        return self._get_local_client().get_with_info(uri, **kwargs)

    def get(self, *args, **kwargs):
        data, _ = self.get_with_info(*args, **kwargs)
        return data

    def list(self, *args, **kwargs):
        client = self._get_local_client()
        return client.list(*args, **kwargs)
    
    Get = get

    GetAndUpdate = get_and_update = functools.partialmethod(
        get, update_cache=True)


if __name__ == '__main__':
    # Example usage
    # client = Client(conf_path='/root/codespace/xpuyu/petreloss.conf')
    # ########################### list
    # data = client.list('xsky:s3://st2pj/20250302/images/public-video/TVQA_frame/s09e08_seg02_clip_08/')
    # def extract_frame_number(filename):
    #     # Extract the numeric part from the filename using regular expressions
    #     match = re.search(r'_(\d+).jpg$', filename)
    #     return int(match.group(1)) if match else -1


    # def sort_frames(frame_paths):
    #     # Extract filenames from each path and sort by their numeric part
    #     return sorted(frame_paths, key=lambda x: extract_frame_number(os.path.basename(x)))
    # print(len(sort_frames(data)))  #

    # s3backend debug
    s3backend = S3Backend(
        end_point_url='http://xceph-outside.pjlab.org.cn:8060',  # Replace with your S3 endpoint
        access_key_id='cX0zA9eC0uO3eL3nZ3uW',  # Replace with your access key ID
        secret_access_key='hG8vO3cW7iL2nM7gP3lP6tE4eY8kZ5'  # Replace with your secret access key
    )
    filepath = 's3://internvl2/datasets/ai2diagram/ai2d/abc_images/4109.png'
    data = s3backend.get(filepath)
    print(data[:100])  # Print first 100 bytes of the data
    