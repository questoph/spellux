# Script to auto-normalize xml files from the STRIPS project

import spellux
import os
import glob
import argparse
import xml.etree.ElementTree as ET
from progressbar import ProgressBar
pbar = ProgressBar()

thedir = os.path.dirname(__file__)

# Define argument parser to pass args with script
parser = argparse.ArgumentParser('Auto-normalize xml files')
parser.add_argument("--indir", type=str, default=thedir, help='Specify the path to the directory')
parser.add_argument("--tag", type=str, default='w', help='Specify the xml tag to process (default= "w")')
parser.add_argument("--outdir", type=str, required=True, help='Specify the path to the saving directory')
args = parser.parse_args()

# Set list of files for normalization
xmldir = args.indir
ending = "*.xml"
path = os.path.join(str(xmldir), ending)
files = glob.glob(path)
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
        # Skip empty elements (entry.text is None for <w/> tags)
        if text is None or text.strip() == "":
            continue
        corrected = spellux.normalize_text(text, mode="model", add_matches=True, sim_ratio=0.8, stats=False, print_unknown=False, nrule=False, indexing=False, lemmatize=True, stopwords=False, output="json", progress=False)
        # doc_text is empty if the entry held no correctable token
        if not corrected:
            continue
        entry.set("corr", corrected[0]["correction"])
        entry.set("lemma", corrected[0]["lemma"])

    # Build the output path from the base filename, independent of input path depth
    base, suff = os.path.splitext(os.path.basename(file))
    outpath = os.path.join(args.outdir, base + "_normalized" + suff)
    tree.write(outpath)
    print("File done!\n")

print("All files corrected!")
