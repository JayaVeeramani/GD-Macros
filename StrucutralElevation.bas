Attribute VB_Name = "StrucutralElevation"
Public swApp As SldWorks.SldWorks
Public swTopLevelModel As SldWorks.ModelDoc2
Public swConfig As SldWorks.Configuration
Public IsSubAssyFormClicked As Boolean

Sub main()

    Set swApp = Application.SldWorks
    Set swTopLevelModel = swApp.ActiveDoc
    
    If swTopLevelModel.GetType = swDocumentTypes_e.swDocASSEMBLY Then

        Call PopulateDisplayStateInForm
        WallDrawingForm.ProjectNoBox.Value = Mid(swTopLevelModel.GetPathName, InStrRev(swTopLevelModel.GetPathName, "\") + 1, 6)
        WallDrawingForm.Show vbModeless

    End If
    
End Sub

Function PopulateDisplayStateInForm()

    Set swConfig = swTopLevelModel.ConfigurationManager.ActiveConfiguration
    
    Dim vDisplayStateNames As Variant
    vDisplayStateNames = swConfig.GetDisplayStates()
    
    Dim i As Integer
    For i = LBound(vDisplayStateNames) To UBound(vDisplayStateNames)
        
        With WallDrawingForm.DisplayList
        
            .AddItem
            .List(i) = vDisplayStateNames(i)
            
        End With
        
    Next i
    
End Function


Sub GetMaxMinPoint(LowerBoundPoint As Variant, HigherBoundPoint As Variant, _
            ByRef OutLower As Double, ByRef OutHigher As Double)

    If LowerBoundPoint < HigherBoundPoint Then
    
        OutLower = LowerBoundPoint
        OutHigher = HigherBoundPoint
        
    Else

        OutLower = HigherBoundPoint
        OutHigher = LowerBoundPoint
        
    End If
    
End Sub


            


