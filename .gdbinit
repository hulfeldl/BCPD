python
import sys
sys.path.insert(0, '//home/hulfeldl/Documents/10_Code/BCPD/.pretty-printers/')
from printers import register_eigen_printers
register_eigen_printers(None)
end