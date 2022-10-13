For the full code visit [github.com/](https://www.github.com/).

# Package Layout

## [Datatypes/](datatypes/index.md) 
Contains modules with objects to hold:

1. configuration settings ([config.py](datatypes/config.md))
2. measurement data ([data.py](datatypes/data.md))
3. spectral data ([spectral.py](datatypes/spectral.md))
    
## [Models/](models/index.md) 
Contains lmfit Models and CompositeModels for generating and/or fitting:

1. periodic functions ([periodic.py](models/periodic.md))
2. etalon interference ([etalon.py](models/etalon.md))
3. spectral lineshapes and superpositions of lines ([spectral.py](models/spectral.md))
4. spectral absorption and transmission ([transmission.py](models/transmission.md))
5. utility functions for the other models modules ([utilities.py](models/utilities.md))
    
## [Processing/](processing/index.md)             
Contains helper functions for (pre-) processing data

1. helps index data objects ([indexing.py](processing/indexing.md))
2. builds filters for wms ([filtering.py](processing/filtering.md))
3. etalon specific helpers ([etalon/](processing/etalon/index.md))
    * to hold etalon derived data ([etalondata.py](processing/etalon/etalondata.md))
    * to fit intensity bounds ([intensityfitting.py](processing/etalon/etalondata.md))  
    * to find interference peaks ([peakfinding.py](processing/etalon/peakfinding.md))
    * to identify invalid peaks ([turnfinding.py](processing/etalon/turnfinding.md))
        
## [wms.py](wms.md)                  
Objects for holding wms data, performing calculations, and plotting 

## [utilities.py](utilities.md)            
Contains helper functions 
