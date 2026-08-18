# Paper

The associated open-access article is:

> Chinh Tan Ly and Quang Linh Huynh, "Human Behavior Recognition Using 2D Time-Series Data and Machine Learning Models," *Journal of Physics: Conference Series*, vol. 3180, 012004, 2026. DOI: [10.1088/1742-6596/3180/1/012004](https://doi.org/10.1088/1742-6596/3180/1/012004).

`Ly_2026_J_Phys_Conf_Ser_3180_012004_corrected_author_copy.pdf` is an author-corrected copy. It is not a publisher-issued correction and must not be described as an updated IOP version.

The only visual change is Figure 1 on PDF page 4. The published PDF accidentally repeated the "Segment pose sequences into 5s windows" stage. The corrected figure contains one segmentation stage and presents the intended seven-stage pipeline. All other article pages and text are preserved.

The correction can be reproduced with:

```powershell
python tools/fix_paper_figure1.py `
  path/to/Ly_2026_J._Phys.__Conf._Ser._3180_012004.pdf `
  paper/Ly_2026_J_Phys_Conf_Ser_3180_012004_corrected_author_copy.pdf
```

The article is published under the Creative Commons Attribution 4.0 license. Attribution and the original DOI must be retained when redistributing the author-corrected copy.
