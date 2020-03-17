# Script to auto-normalize xml files from the STRIPS project

import spellux
import glob
import argparse
import xml.etree.ElementTree as ET
from progressbar import ProgressBar
pbar = ProgressBar()


# Define argument parser to pass args with script
parser = argparse.ArgumentParser('Auto-normalize xml files')
parser.add_argument("--dir", type=str, help='Specify the path to the directory')
parser.add_argument("--tag", type=str, default='w', help='Specify the xml tag to process (default= "w")')
args = parser.parse_args()

# Set list of files for normalization
files = glob.glob(args.dir + "/*.xml")
print("\nNumber of files to normalize: {}" .format(str(len(files))))

# Start correction routine
for file in files:
    print("\nCorrecting: " + file)
    
    # Parse xml file and structure
    tree = ET.parse(file)
    root = tree.getroot()

    # Normalize each entry on word level
    for entry in pbar(root.findall('.//' + args.tag)).start():
        text = entry.text
        corrected = spellux.normalize_text(text, mode="model", add_matches=True, sim_ratio=75, stats=False, print_unknown=False, nrule=False, indexing=False, lemmatize=False, tolist=False, progress=False)
        entry.set("corr", corrected)
    
    name = file[4:-4] + "_normalized"
    suff = file[-4:]
    tree.write(args.dir + "normalized/" + name + suff)
    print("File done!\n")

spellux.global_stats()
print("All files corrected!")