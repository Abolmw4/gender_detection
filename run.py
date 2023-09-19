from classifier import GenderSeperator
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Gender Seperator parameter")
    parser.add_argument('-i', '--input', help="Input directory absolute path")
    parser.add_argument('-o', '--output', help="Output directory absolute path")
    parser.add_argument('-m', '--model', default='./Abolfazl_gender_detection.pt', help='model absolute path')
    parser.add_argument('-d', '--device', default='cuda', help="your device cpu or cuda")
    args = parser.parse_args()

    test = GenderSeperator(args.input, args.output, args.model)
    print(test)
    test(device=args.device)


if __name__ == "__main__":
    main()
