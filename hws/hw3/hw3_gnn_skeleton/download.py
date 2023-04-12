#
import argparse
import os
import requests # type: ignore[import]



def ddi(path: str, /) -> None:
    R"""
    Prepare DDI dataset.
    """
    #
    print("Downloading DDI raw data ...")
    src = (
        "https://raw.githubusercontent.com/gao462/DrugDrugInteraction/main/ddi"
        ".npy"
    )
    tar = os.path.join(path, "ddi.npy")
    if os.path.isfile(tar):
        #
        return
    remote = requests.get(src)
    with open(tar, "wb") as file:
        #
        file.write(remote.content)



def main(*ARGS) -> None:
    R"""
    Main execution.
    """
    #
    parser = argparse.ArgumentParser(description="Download Execution")
    parser.add_argument(
        "--source",
        type=str, required=False, default="data",
        help="Source root directory for data.",
    )
    parser.add_argument("--ddi", action="store_true", help="Prepare DDI data.")
    args = parser.parse_args() if len(ARGS) == 0 else parser.parse_args(ARGS)

    #
    root = args.source
    do_ddi = args.ddi

    #
    if not os.path.isdir(root):
        #
        os.makedirs(root, exist_ok=True)

    #
    if do_ddi:
        #
        ddi(root)


#
if __name__ == "__main__":
    #
    main()