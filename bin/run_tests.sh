cd /Users/hadinayebi/CodingProjects/plsx/
coverage run ./PLSx/dataloader/test/test_dataloader.py
coverage run -a ./PLSx/dataloader/test/test_utils.py
coverage run -a ./PLSx/database/test/test_sqlitedb.py
coverage report -m
coverage html