# Edgespotter Deployment
## Custom character maps

This is only relevant for models trained for different character sets.

If you are trying to convert a model you've trained yourself for some language other than English, which is supported by default by the base model, you'll need to adjust the vocabulary settings.

### Create an alphabet

You'll need to create a file named "lang-chars-#.txt" in directory './char-dicts' where "lang" is the name of the language and "#" is the total number of characters in the language's alphabet. The file should contain a literal string, comprised of the langauge's characters. 

Once the text file with the alphabet is ready, run the `make_dicts.py` script in the projects root:

```bash
python make_dicts.py
```

This will rebuild all alphabet files and create corresponding '.cmap' files, which are simply pickled lists of characters.

### Adjust `VOC_SIZE`

You will also need to adjust the `MODEL.TRANSFORMER.VOC_SIZE` setting. 

The best way to do that is to create a new config file, by copying the file 
'./configs/Base_det_export.yaml' under a descriptive name and adding a line in the 
`MODEL.TRANSFORMER` block (between the lines 11 and 33):

```
VOC_SIZE = 37
```

setting it to whatever number of characters is in your language's alphabet.

## Running inference

Check out 'infer_image.py' for an example of how to perform inference on a single image. You can also use 'demo.py' for video streaming inference.

## Acknowledgement
* The code is implemented based on *[deepSolo-onnx](https://github.com/agoryuno/deepsolo-onnx)*. We would like to express our sincere thanks to the contributors.


 