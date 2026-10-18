from pathlib import Path
import re
import xml.etree.ElementTree as ET
from config import DATA_DIR, LANGUAGES

#currently parses 

'''
format of xml files
<?xml version="1.0" ?>
<cesDoc version="4">
        <cesHeader version="2">
                <fileDesc>
                        <titleStmt>
                                <h.title>
                                        Bible(ENG)
                                </h.title>
                                <respStmt>
                                        <respType>
                                                Constructed in accordance to CES standards.
                                        </respType>
                                        <respName>
                                                Christos Christodoulopoulos
                                        </respName>
                                </respStmt>
                        </titleStmt>
                        <extent>
                                <wordCount>
                                        789637
                                </wordCount>
                                <byteCount units="bytes">
                                        5412199
                                </byteCount>
                        </extent>
                        <publicationStmt>
                                <distributor>
                                        Bible Database
                                </distributor>
                                <eAddress>
                                        http://bibledatabase.com
                                </eAddress>
                                <availability>
                                        free
                                </availability>
                        </publicationStmt>
                        <sourceDesc/>
                </fileDesc>
                <encodingDesc>
                        <projectDesc>
                                This is a part of multi-lingual corpus of the bible.
                        </projectDesc>
                        <editorialDesc>
                                <conformance level="1">
                                        Corpus Encoding Standard, Version 2.0
                                </conformance>
                                <correction method="silent" status="medium"/>
                                <segmentation>
                                        Marked up to the level of chapter and verse.
                                </segmentation>
                        </editorialDesc>
                        <tagsDecl/>
                </encodingDesc>
                <profileDesc>
                        <langUsage>
                                <language id="en" iso639="eng">
                                        English
                                </language>
                        </langUsage>
                        <wsdUsage>
                                <writingSystem id="utf-8"/>
                        </wsdUsage>
                </profileDesc>
        </cesHeader>
        <text>
                <body id="Bible" lang="en">
                        <div id="b.GEN" type="book">
                                <div id="b.GEN.1" type="chapter">
                                        <seg id="b.GEN.1.1" type="verse">
                                                In the beginning God cr(venv) 
'''


def parse_bible_xml(xml_file):
    """
    Parses a Bible XML file and returns a list of verses.
    Each verse is represented as a dictionary with keys: 'book', 'chapter', 'verse', 'text'.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    verses = []
    currently_reading_the_new_testament = False
    for seg in root.iter("seg"):
        if seg.get("type") != "verse":
            continue
        _, book, chapter, verse = seg.get("id").split(".", 3)      
        if book == "MAT":
            currently_reading_the_new_testament = True
        if not currently_reading_the_new_testament:
            continue
        text = re.sub(r"\s+", " ", "".join(seg.itertext())).strip()  
        verses.append({"book": book, "chapter": chapter, "verse": verse, "text": text})
    return verses


def is_non_empty_verse(verse):
    """
    Checks if a verse dictionary has non-empty text.
    """
    return bool(verse.get('text', '').strip())

def read_bible():
    """
    Reads all Bible XML files in the DATA_DIR and returns a dictionary mapping language codes to lists of verses.
    Each verse is represented as a dictionary with keys: 'book', 'chapter', 'verse', 'text'.
    """
    corpus = {}
    for code, fname in LANGUAGES.items():
        xml_file = DATA_DIR / "bible-corpus" / "bibles" / fname
        if xml_file.exists():
            verses = parse_bible_xml(xml_file)
            verses = [v for v in verses if is_non_empty_verse(v)]
            corpus[code] = verses
        else:
            print(f"Warning: {xml_file} does not exist.")
            corpus[code] = []
    return corpus

 
if __name__ == "__main__":
    texts = read_bible()
    for lang, verses in texts.items():
        print(lang, len(verses))



