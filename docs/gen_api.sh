# Generates API html for horton_part while ignoring the test and data folders.
# Stores it in pyapi/

sphinx-apidoc -a -o pyapi/ ../src/horton_part ../src/horton_part/tests/ ../src/horton_part/test/ ../src/horton_part/data/ --separate
