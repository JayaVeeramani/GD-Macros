Attribute VB_Name = "Update_Gauge"
Option Explicit

Sub main()

    Dim swApp As SldWorks.SldWorks
    Set swApp = Application.SldWorks
    
    Dim swModel As IModelDoc2
    Set swModel = swApp.ActiveDoc

    If swModel.GetType = swDocumentTypes_e.swDocASSEMBLY Then
        
        Dim swAssyDoc As AssemblyDoc
        Set swAssyDoc = NewAssemblyDoc(swModel)
        
        Dim partDict As Object
        Set partDict = swAssyDoc.GetUniquePartDoc
        
        Call userFormFunctions.PopulateProfileList(partDict)

        UpdateGaugeForm.Show vbModeless
        
    Else
    
        MsgBox "Please Open Assembly Environment"
    
    End If
        
End Sub








