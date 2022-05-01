cd /Users/hadinayebi/CodingProjects/plsx/PLSx/
coverage run ./dataloader/test/test_dataloader.py
coverage run -a ./database/test/test_sqlitedb.py
coverage report -m
coverage html