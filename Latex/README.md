# Thesis Template

This is a LaTeX template for thesis projects of the Distributed Systems group at the University of Duisburg-Essen.

## Usage

**Please read the [main file](thesis.tex) in its entirety.**
The options and packages used are explained there.
You can change it to better fit your needs.

_Also read the included example PDF
as it contains some additional information on usage
and resources for learning LaTeX._

Organize your chapters inside the _content_ folder and include the files in _thesis.tex_.
This makes it easier to keep track of smaller changes and single parts of your thesis.
Make sure to properly edit the [front page](frontpage.tex).  
Do _not_ edit the _clause.tex_ (except for the name and location) but remember to sign it before submitting your work.
Also remember to set the correct metadata in the main file and comment out the color changes made to references for printing.  
You can use the `\todo{}` command to include easily visible comments in the document.

## Requirements

The easiest way to ensure all required packages are installed is to utilize a preconfigured [Docker](https://docs.docker.com/get-docker/) container like [this one](https://github.com/dante-ev/docker-texlive).  
However, it should also work with locally installed tools and in various common LaTeX editors.

### Manual

If [GNU make](https://www.gnu.org/software/make/) is available on your system, you can simply run `make setup` and `make` to compile your documents.
Make is commonly available on macOS and Linux, and it is possible to install on Windows: [further information](https://github.com/mbuilov/gnumake-windows)  
You can also just invoke the required commands directly.
To see these, just have a look at the [Makefile](Makefile).
Its structure is simple, the required commands are written in the lines starting with tabs.

### Automated

This template can also be used in combination with tools like [Visual Studio Code](https://code.visualstudio.com/) and the [LaTeX Workshop extension](https://github.com/James-Yu/LaTeX-Workshop).
It should work out of the box.  
If you wish to uitilize the Docker container, all you need to do is add the following two options to your VS Code settings:
```
    "latex-workshop.docker.enabled": true,
    "latex-workshop.docker.image.latex": "danteev/texlive",
```

## Additional Information

It is useful to utilize a spell checking tool like [cSpell](https://marketplace.visualstudio.com/items?itemName=streetsidesoftware.code-spell-checker) in order to catch the most blatant errors that might occur when writing for prolonged periods.
