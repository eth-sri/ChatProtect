### What is this?

This directory contains a number of outputs by gLMs to test new procedures against.
This is necessary to make different processing approaches comparable (i.e. triple extraction, sentence extraction, contradiction detection)
as consecutive runs of gLMs usually produce different results.

The outputs are structured the same for the three test sets `validation`, `test`, and `test_big`.
`entities.txt` at the top level of each set contains the entities covered by the test set.

### How is it structured?

The top level directory contains loosely described categories of outputs.

1. descriptions - Generic descriptions of entities obtained by prompting "Tell me about X"
1. sentences - Sentences about things
1. decisions - A log of decisions made when revising descriptions
1. new_descriptions - Revised descriptions of entities

Each directory contains several subdirectories that are first denote aLM, then gLM and then the entity/prompt used to generate the output.
The file names on the lowest level do not contain any semantic meaning but should be nonempty, not contain any non-printable characters, space characters (tab, space, newline) or special characters.
The files are json-formatted and contain fields depending on their category.

`sentences` maps to the `ExtSentence` class, `descriptions` and `new_descriptions` map to `Description` and `decisions` maps to `Decisions` in `chatprotect.util`.

They are optionally tagged for use as ground truth (i.e. in the case of `test` and `validation`)
or with the original tag produced during sentence generation by the aLM.
