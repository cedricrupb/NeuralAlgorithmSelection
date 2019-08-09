import argparse
import lmdb
import os
import shutil
import zipfile


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            root_ = os.path.basename(os.path.normpath(root))
            print(root_)
            ziph.write(os.path.join(root, file), arcname=os.path.join(root_,
                                                                      file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("source")

    args = parser.parse_args()

    src = args.source
    tmp = os.path.abspath(os.path.join(src, "..", "tmp"))
    base = os.path.basename(os.path.normpath(src))
    target = os.path.abspath(os.path.join(src, "..", base+".zip"))

    os.makedirs(tmp)

    print("Copying %s..." % src)
    db = lmdb.open(src)
    db.copy(tmp, compact=True)

    print("Deleting %s..." % src)
    shutil.rmtree(src)

    print("Moving tmp...")
    shutil.move(tmp, src)

    print("Zip to %s" % target)
    zipf = zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED)
    zipdir(src, zipf)
    zipf.close()
