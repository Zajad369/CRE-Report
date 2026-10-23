# KTU Laboratory Work Report Template

An editable LaTeX template for KTU laboratory work reports.

## Getting started

1. Open the repository folder in VS Code.
2. Edit `variables.tex` with the information for your report.
3. Replace the instructional text in `chapters/report_content.tex` and
   `chapters/conclusions.tex` with your report content.
4. Add references to `references.bib` and cite them in the report.
5. Install the recommended **LaTeX Workshop** extension if VS Code asks.
6. Open `main.tex` and save it. The project builds automatically with MiKTeX
   and `latexmk`; generated files are placed in `build/`.
7. Open the PDF using the LaTeX Workshop PDF button or `Ctrl+Alt+V`.

## Where to edit

- `variables.tex`: module, title, student, supervisor, city, and year
- `TitlePage.tex`: cover-page layout
- `chapters/report_content.tex`: introduction, theory, methods, results, and
  discussion
- `chapters/conclusions.tex`: conclusions and recommendations
- `references.bib`: bibliography entries

Keep section headings if you want them to appear in the table of contents.
Tables and figures are added in the chapter where they are discussed; their
captions automatically appear in the corresponding lists.

## Requirements

- [MiKTeX](https://miktex.org/download) with `pdflatex`, `latexmk`, and
  `bibtex` available on `PATH`
- Visual Studio Code
- The VS Code extension `James-Yu.latex-workshop`

MiKTeX should be configured to install missing packages automatically. On the
first build, MiKTeX may download packages and fonts; this can take a few
minutes.
