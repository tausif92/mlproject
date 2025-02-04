from setuptools import find_packages, setup

HYPHEN_E_DOT = '-e .'


def get_requirements(file_path):
    '''
    This function will return the list of requirements.
    '''
    with open(file_path) as file_obj:
        lines = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in lines]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements


setup(
    name='mlproject',
    version='0.0.1',
    author='Tausif',
    author_email='tb@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
