cd /Users/hadinayebi/CodingProjects/plsx/
coverage run ./PLSx/utils/test/test_template.py
coverage run -a ./PLSx/dataloader/test/test_dataloader.py
coverage run -a ./PLSx/database/test/test_sqlitedb.py
coverage run -a ./PLSx/dataloader/test/test_utils.py
coverage run -a ./PLSx/autoencoder/test/test_architecture.py
coverage run -a ./PLSx/utils/test/test_custom_argparser.py
coverage run -a ./PLSx/autoencoder/test/test_autoencoder.py
coverage report -m
coverage html