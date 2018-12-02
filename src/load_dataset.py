"""
scikit-learn mimick fetch data set

001C_1187_DATA

"""

from os import environ, listdir, makedirs
from os.path import dirname, exists, expanduser, isdir, join, splitext
import shutil

DEFUALT_DATA_DIRECTORY_ROOT = '001C_1187_DATA'


def get_data_home(data_home=None):
	"""

	Return the path of our data dir.
	This folder is used by some large dataset loaders to avoid downloading the
	data several times.
	By default the data dir is set to a folder named '001C_1187_lab7_data' in the
	user home folder.
	Alternatively, it can be set by the '001C_1187_DATA' environment
	variable or programmatically by giving an explicit folder path. The '~'
	symbol is expanded to the user home folder.
	If the folder does not already exist, it is automatically created.
	Parameters
	----------
	data_home : str | None
		The path to 001C-1187 lab7 data dir.
	"""
	if data_home is None:
		data_home = environ.get('001C_1187_DATA',
								join('~', '001C_1187_lab7_data'))
	data_home = expanduser(data_home)
	if not exists(data_home):
		makedirs(data_home)
	return data_home

def clear_data_home(data_home=None):
    """Delete all the content of the data home cache.
    Parameters
    ----------
    data_home : str | None
        The path to scikit-learn data dir.
    """
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)

# testing
if __name__ == "__main__":

	#print(get_data_home())

	# or
	from sklearn.datasets import load_files

	categories = ['bbcsport.rugby, bbcsport.cricket']
	data_train = load_files(container_path='../data/bbcsport', decode_error='ignore', encoding='utf-8')


	print(data_train.target_names)