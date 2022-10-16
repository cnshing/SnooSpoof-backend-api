Parsing component responsible for converting text generated results into JSON.

Since the initial generated text will eventually be processed and passed through to the frontend, the
parser can be a potential point of failure. Specifically, if the preprocessing fails, then the webpage
can no longer display the generated text regarldess of how well the model or webpage was designed.
Even flaws in processing that introduces some form of biases in the text can be argued as not a true 
impression/representation of the user.

In short a poorly implemented parser means an more inconsistent and inaccurate generated text.
Seperating the parser into its component allows us to properly isolate and test for any
of the above issues.

The text is parsed by extracting the slices of text between each tag. Since slicing is
determined by a corresponding start and end position, text that follows the correct order
of tags will have their index positions to also be ordered from least to greatest:


tag1: str1 tag2: str2 tag3: str3
1    6    12   17   22     28   33
tag1 |----|> (6, 12)
tag2            |----|> (17,22)
tag3                       |----|> (28,33)
6 < 12 < 17 < 22 < 28 < 33 => text properly follows tag1,tag2,tag3 order


tag1: str1 tag3: str3 tag2: str2
1    6    12   17   22     28   33
tag1 |---------------|> (6, 22)
tag2      <|---------------| (28, 12)
tag3            |----------------|> (17, 33)
6 < 22 < 28 <! 12 <? 17 < 33 => tag2 is somehow misordered

Since how we order the tags strictly determines what is being generated, misordered text most likely means something we don't want to be generated is somehow being generated and constitutes invalid text. 













