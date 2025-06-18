Attribute VB_Name = "ObjectInit"
Option Explicit

Function NewPartDoc(swModel As SldWorks.ModelDoc2, ConfigName As String) As PartDoc

    Set NewPartDoc = New PartDoc
    NewPartDoc.Initialize swModel, ConfigName

End Function

Function NewAssemblyDoc(swModel As IModelDoc2) As AssemblyDoc

    Set NewAssemblyDoc = New AssemblyDoc
    NewAssemblyDoc.Initialize swModel

End Function

Function NewISheetMetal(Part As PartDoc) As ISheetMetal

    Set NewISheetMetal = New ISheetMetal
    NewISheetMetal.Initialize Part

End Function



