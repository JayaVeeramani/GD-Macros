Attribute VB_Name = "Module1"

Dim swApp As SldWorks.SldWorks
Dim swDrawing As SldWorks.DrawingDoc


Sub Main()
    
    Set swApp = Application.SldWorks
    Set swDrawing = swApp.ActiveDoc
    
    Call UpdateSheetScales
    
    
End Sub

Sub UpdateSheetScales()
    

    Dim swSelect As SldWorks.SelectionMgr
    Set swSelect = swDrawing.SelectionManager()

    swDrawing.EditTemplate

    Dim SheetFormatName As String
    SheetFormatName = swSheet.GetSheetFormatName
    Dim BoolStatus As Boolean

    Dim swNote As INote
    BoolStatus = swDrawing.Extension.SelectByID2("DetailItem1198@" & SheetFormatName, "NOTE", 0.355252206998469, 3.32049059009041E-02, 0, False, 0, Nothing, 0)
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

Sub UpdateViewScales()

    Dim swSheet As SldWorks.Sheet
    Set swSheet = swDrawing.GetCurrentSheet

    Dim swSelect As SldWorks.SelectionMgr
    Set swSelect = swDrawing.SelectionManager()

    swDrawing.EditTemplate

    Dim SheetFormatName As String
    SheetFormatName = swSheet.GetSheetFormatName
    Dim BoolStatus As Boolean

    Dim swNote As INote
    BoolStatus = swDrawing.Extension.SelectByID2("DetailItem1198@" & SheetFormatName, "NOTE", 0.355252206998469, 3.32049059009041E-02, 0, False, 0, Nothing, 0)
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
