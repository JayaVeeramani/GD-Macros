VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "PostRegenerateListener"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False

Dim WithEvents swApp As SldWorks.SldWorks
Attribute swApp.VB_VarHelpID = -1
Dim WithEvents swModel As SldWorks.DrawingDoc
Attribute swModel.VB_VarHelpID = -1
Dim WithEvents swDrawing As SldWorks.DrawingDoc



Dim cScaleRatio As Double


Sub Init(app As SldWorks.SldWorks, model As SldWorks.ModelDoc2)
    
    Set swApp = app
    Set swDrawing = model
    

End Sub

Private Function swApp_ActiveDocChangeNotify() As Long
        Call UpdateSheetScale
End Function

Private Function swApp_CommandOpenPreNotify(ByVal Command As Long, ByVal UserCommand As Long) As Long

    Debug.Print reason
    Debug.Print Command

    If (Command = 2) Or Command = swCommands_Print Then

        Call UpdateSheetScale
        
        
    End If

End Function


Sub UpdateSheetScale()

    Dim swSheet As SldWorks.Sheet
    Set swSheet = swDrawing.GetCurrentSheet
    
    Dim swSelect As SldWorks.SelectionMgr
    Set swSelect = swDrawing.SelectionManager()

    swDrawing.EditTemplate

    Dim SheetFormatName As String
    SheetFormatName = swSheet.GetSheetFormatName
    Dim BoolStatus As Boolean

    Dim swNote As INote
    BoolStatus = swDrawing.Extension.SelectByID2("DetailItem1222@" & SheetFormatName, "NOTE", 0.355252206998469, 3.32049059009041E-02, 0, False, 0, Nothing, 0)
    Set swNote = swSelect.GetSelectedObject6(1, -1)

    Dim vSheetProps As Variant
    vSheetProps = swSheet.GetProperties

    Dim Numerator As Double
    Dim Denominator As Double

    Numerator = vSheetProps(2)
    Denominator = vSheetProps(3)

    Dim ArcScale As Double
    ArcScale = 12 * (Numerator / Denominator)

    Dim ScaleText As String

    If ArcScale = 12 Then

        ScaleText = "1'=1'"

    Else
        
        Dim Remainder As Double
        Remainder = ArcScale - Int(ArcScale)

        
        If Remainder = 0 Then
            
            ScaleText = ArcScale & Chr(34) & "=1'"
        
        Else

            Dim swUserUnits As SldWorks.UserUnit
            Set swUserUnits = swApp.GetUserUnit(swUserUnitsType_e.swLengthUnit)
    
            swUserUnits.FractionBase = swFractionDisplay_e.swFRACTION
            swUserUnits.SpecificUnitType = swLengthUnit_e.swINCHES
    
            swUserUnits.RoundToFraction = True
            swUserUnits.FractionValue = 64

            ScaleText = swUserUnits.ConvertToUserUnit(ArcScale * 0.0254, True, True)

            
            ScaleText = ScaleText & "=1'"
            
        End If

    End If

    swNote.SetText (ScaleText)
    
    
    swDrawing.EditSheet


End Sub

Private Function swApp_PropertySheetCreateNotify(ByVal Sheet As Object, ByVal sheetType As Long) As Long

End Function

Private Function swDrawing_RegenNotify() As Long

End Function




Private Function GetSheetScale() As Double

    Dim swSheet As SldWorks.Sheet
    Set swSheet = swDrawing.GetCurrentSheet
    
    Dim vProps  As Variant
    vProps = swSheet.GetProperties
    
    GetSheetScale = vProps(2) / vProps(3)
    
End Function
