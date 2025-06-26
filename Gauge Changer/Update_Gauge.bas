Attribute VB_Name = "Update_Gauge"
Option Explicit

Public swModel As SldWorks.ModelDoc2
Public swApp As SldWorks.SldWorks
Public swAssyDoc As AssemblyDoc
Public partDict As Scripting.Dictionary
'Public NewDSName As String
'Public OldDsName As String


Sub main()

    Set swApp = Application.SldWorks
    Set swModel = swApp.ActiveDoc

    If swModel.GetType = swDocumentTypes_e.swDocASSEMBLY Then
        
        Set swAssyDoc = NewAssemblyDoc(swModel)
        Set partDict = swAssyDoc.GetUniquePartDoc
        
        Call userFormFunctions.PopulateProfileList(partDict)
        
        'OldDsName = swAssyDoc.GetCurrentDs
        
        UpdateGaugeForm.Show vbModeless
        
    Else
    
        MsgBox "Please Open Assembly Environment"
    
    End If
        
End Sub








