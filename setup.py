import setuptools

readme = open('README.md').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')
requirements = open('requirements.txt').read().splitlines()

setuptools.setup(
    name='DirectDmTargets',
    version='0.4.0',
    description='Probing the complementarity of several targets used in '
                'Direct Detection Experiments for Dark Matter',
    long_description=readme + '\n\n' + history,
    author='Joran Angevaare',
    url='https://github.com/jorana/DD_DM_targets',
    packages=setuptools.find_packages(),
    setup_requires=['pytest-runner'],
    install_requires=requirements,
    package_dir={'DirectDmTargets': 'DirectDmTargets'},
    package_data={'DirectDmTargets': [
        'data/*']},
    tests_require=requirements + ['pytest',
                                  'hypothesis-numpy'],
    scripts=['scripts/run_combined_multinest',
             'scripts/run_dddm_emcee',
             'scripts/run_dddm_multinest',
             ],
    keywords='todo',
    classifiers=['Intended Audience :: Science/Research',
                 'Development Status :: 3 - Alpha',
                 'Programming Language :: Python',
                 'Programming Language :: Python :: 3'],
    zip_safe=False)
