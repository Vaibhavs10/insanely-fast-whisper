import json
import argparse

def format_seconds(seconds):
    whole_seconds = int(seconds)
    milliseconds = int((seconds - whole_seconds) * 1000)

    hours = whole_seconds // 3600
    minutes = (whole_seconds % 3600) // 60
    seconds = whole_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def convert_to_srt(input_path, output_path, verbose):
    with open(input_path, 'r') as file:
        data = json.load(file)

    rst_string = ''
    for index, chunk in enumerate(data['chunks'], 1):
        text = chunk['text']
        start, end = chunk['timestamp'][0], chunk['timestamp'][1]
        start_format, end_format = format_seconds(start), format_seconds(end)
        srt_entry = f"{index}\n{start_format} --> {end_format}\n{text}\n\n"
        
        if verbose:
            print(srt_entry)
        
        rst_string += srt_entry

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(rst_string)

def main():
    parser = argparse.ArgumentParser(description="Convert JSON to SRT format.")
    parser.add_argument("input_file", help="Input JSON file path")
    parser.add_argument("-o", "--output_file", default="output.srt", help="Output SRT file path (default: output.srt)")
    parser.add_argument("--verbose", action="store_true", help="Print each SRT entry as it's added")

    args = parser.parse_args()
    convert_to_srt(args.input_file, args.output_file, args.verbose)

if __name__ == "__main__":
    # Example Usage: 
    # python convert_srt.py output.json -o my_caption.srt
    main()
