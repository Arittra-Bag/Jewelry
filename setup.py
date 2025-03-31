from setuptools import setup

setup(
    name="jewelry_shop_security",
    version="1.0",
    packages=["jewelry_shop_security"],
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
        "Pillow>=8.0.0",
        "tkcalendar>=1.6.1",
    ],
    entry_points={
        'console_scripts': [
            'jewelry_shop_security=jewelry_shop_security.dashboard:main',
        ],
    }
) 