Attribute VB_Name = "Factory"

Public Function NewPartObject(Part As SldWorks.PartDoc) As IPart

    Set NewPartObject = New IPart
    NewPartObject.Init Part
    
End Function

Public Function NewCutlistObject(Feat As SldWorks.Feature) As ICutlist

    Set NewCutlistObject = New ICutlist
    NewCutlistObject.Init Feat
    
End Function
