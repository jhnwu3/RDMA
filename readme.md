# RDMA - Rare Disease Mining Agents


To start, please:
    
    pip install -r requirements.txt


Download these prerequisite files for the embedded documents [here.](https://drive.google.com/file/d/16wpcexHf2KDZ4w2qBHrTp8dn1oa59ABM/view?usp=sharing)

Make sure to unzip the files and place them in a location where you can reference their pathing.


To see how to use RDMA, we have provided a jupyter notebook:

    example.ipynb

## Using the provided annotation UI
We note that it is possible to use our existing annotation tool locally. Simply, double click or open annotation_tool.html, and you'll be greeted with this interface below:

![UI_Interface](figs/AnnotationToolUI.png)

Simply click the upload button and upload your .json file. 

![Upload Button](figs/Uploadbutton.png)

Upload your file.

![Upload Button Clicked](figs/UploadButtonAnnotation.drawio.png)


Then, you'll be greeted with the annotation display where you can click next, and declare whether or not an entity is a rare disease or not.

![Annotating](figs/AnnotatingUI.png)

Once you're done, hit the green export button in the top right, it will ask to save a corrections .json file. 

![ExportButton](figs/ExportButton.png)


Some important notes:

**Do not refresh the page or you will lose all of your progress. Do not exist on accident. There's no database or backend that's tracking your annotations.**