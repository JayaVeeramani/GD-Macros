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
Dim WithEvents swDrawing As SldWorks.DrawingDoc
Dim swUserUnits As SldWorks.UserUnit


Sub Init(app As SldWorks.SldWorks, model As SldWorks.ModelDoc2)
    
    Set swApp = app
    Set swDrawing = model
        
    Set swUserUnits = swApp.GetUserUnit(swUserUnitsType_e.swLengthUnit)

    swUserUnits.FractionBase = swFractionDisplay_e.swFRACTION
    swUserUnits.SpecificUnitType = swLengthUnit_e.swINCHES

    swUserUnits.RoundToFraction = True
    swUserUnits.FractionValue = 64
    

End Sub

Private Function swApp_ActiveDocChangeNotify() As Long
        Call UpdateSheetScale
End Function

Private Function swApp_CommandOpenPreNotify(ByVal Command As Long, ByVal UserCommand As Long) As Long

    Debug.Print reason
    Debug.Print Command

    If (Command = swCommands_Save) Or Command = swCommands_Print Or _
    Command = swCommands_SaveAs Or Command = swCommands_SaveAll Then

        Call UpdateSheetScale
        
        
    End If

End Function


Sub UpdateSheetScale()

 Dim swActiveSheet As SldWorks.Sheet
    Set swActiveSheet = swDrawing.GetCurrentSheet
    
    Dim vSheets As Variant
    vSheets = swDrawing.GetSheetNames
    
    Dim swSelect As SldWorks.SelectionMgr
    Set swSelect = swDrawing.SelectionManager()
    
    Dim i As Integer
    For i = LBound(vSheets) To UBound(vSheets)
    
        'swDrawing.EditTemplate
        
        Dim swSheet As SldWorks.Sheet
        Set swSheet = swDrawing.Sheet(vSheets(i))

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
        
        Dim sheetScale As Double
        sheetScale = (Numerator / Denominator)
    
        swNote.SetText (GetArcScaleText(sheetScale))
        Call UpdateViewScaleNotes(swSheet, sheetScale)
        'swDrawing.EditSheet

    Next i
    
    'swDrawing.ActivateSheet swActiveSheet.GetName


End Sub

Private Function UpdateViewScaleNotes(swSheet As SldWorks.Sheet, sheetScale As Double)

    Dim vViews As Variant
    vViews = swSheet.GetViews
    
    If Not IsEmpty(vViews) Then
    
        Dim i As Integer
        For i = LBound(vViews) To UBound(vViews)
        
            Dim swView As SldWorks.View
            Set swView = vViews(i)
            
            If Not (InStr(swView.Name, "*") > 0) Then
            
                Debug.Print swView.Name & swView.Type
                
                If swView.ScaleDecimal <> sheetScale Then
                
                    Dim swNote As SldWorks.Note
                    Set swNote = swView.GetFirstNote
                    
                    If Not swNote Is Nothing Then
                        
                        Dim IsFound As Boolean
                        IsFound = False
                    
                        While Not swNote Is Nothing Or False = IsFound
                        
                            Debug.Print swNote.GetText
                            
                            Dim PropLinkedText As String
                            PropLinkedText = swNote.PropertyLinkedText
                            
                            Debug.Print PropLinkedText

                            If InStr(PropLinkedText, "<VLSCALE>") > 0 Or InStr(UCase(PropLinkedText), "SCALE") > 0 Then
                            
                                IsFound = True
                                
                                Dim ArcScaleText As String
                                ArcScaleText = GetArcScaleText(swView.ScaleDecimal)
                                
                                Dim RegEx As Object
                                Set RegEx = CreateObject("VBScript.RegExp")
                
                                RegEx.Pattern = "(?:\d+(?:\s+\d+/\d+)?|\d+/\d+|\d+)(?:" & Chr(34) & "\s*=\s*\d+'|'\s*=\s*\d+')"
                                RegEx.Global = True
                                 
                                Dim AllMatches As Object
                                Set AllMatches = RegEx.Execute(PropLinkedText)
         
                                If InStr(PropLinkedText, "<VLSCALEV>") > 0 Then
                                    
                                    If AllMatches.Count > 0 Then

                                        PropLinkedText = RegEx.Replace(PropLinkedText, "")
                                    
                                    End If
                                    
                                    PropLinkedText = Replace(PropLinkedText, "<VLSCALEV>", ArcScaleText)
                                    
                                Else
                                    If AllMatches.Count > 0 Then
                                        PropLinkedText = RegEx.Replace(PropLinkedText, ArcScaleText)
                                        
                                    Else
                                    
                                        RegEx.Pattern = "\d+\s*:\s*\d+"
                                        Set AllMatches = RegEx.Execute(PropLinkedText)
                                        PropLinkedText = RegEx.Replace(PropLinkedText, ArcScaleText)
                                        
                                        If AllMatches.Count = 0 Then
                                            
                                            Dim ScalePos As Integer
                                            If InStr(PropLinkedText, "VLSCALE") > 0 Then
                                            
                                                ScalePos = InStrRev(PropLinkedText, "<VLSCALE>") + Len("<VLSCALE>")
                                                Call InsertTextAfterThisPos(PropLinkedText, ScalePos, ArcScaleText)
                                            
                                            Else
                                            
                                                ScalePos = InStr(PropLinkedText, "SCALE") + Len("SCALE")
                                                Call InsertTextAfterThisPos(PropLinkedText, ScalePos, ArcScaleText)
                                                
                                            End If
                                        
                                        
                                        End If
                                        
                                    End If
                                        

                                End If
                                
                                RegEx.Pattern = "(?:<VLSCALE>|SCALE)"
                                Debug.Print PropLinkedText
                                PropLinkedText = RegEx.Replace(PropLinkedText, "")
                                Debug.Print PropLinkedText
                                
                                Dim ScalePosInText  As Integer
                                ScalePosInText = InStr(PropLinkedText, ArcScaleText) - 1
                                
                                PropLinkedText = RTrim(Left(PropLinkedText, ScalePosInText)) & " <VLSCALE> " & LTrim(Right(PropLinkedText, Len(PropLinkedText) - ScalePosInText))
                                swNote.PropertyLinkedText = PropLinkedText
                                Debug.Print PropLinkedText
    
                            End If
    
                            Set swNote = swNote.GetNext
    
                        
                        Wend
                        
                    End If
    
                End If
            
            End If

        Next i

    End If
    
End Function

Private Function GetArcScaleText(sheetScale As Double) As String

    Dim ArcScale As Double
    ArcScale = 12 * sheetScale
    
    Dim ScaleText As String

    If ArcScale = 12 Then

        ScaleText = "1'=1'"

    Else
        
        Dim Remainder As Double
        Remainder = ArcScale - Int(ArcScale)

        
        If Remainder = 0 Then
            
            GetArcScaleText = ArcScale & Chr(34) & "=1'"
        
        Else

            GetArcScaleText = swUserUnits.ConvertToUserUnit(ArcScale * 0.0254, True, True)
            GetArcScaleText = GetArcScaleText & "=1'"
            
        End If

    End If
    
End Function

Private Sub InsertTextAfterThisPos(ByRef OrgText As String, Pos As Integer, TextToInsert As String)
    
    Debug.Print Left(OrgText, Pos)
    Debug.Print Right(OrgText, Len(OrgText) - Pos)
    OrgText = Left(OrgText, Pos - 1) & TextToInsert & Right(OrgText, Len(OrgText) - Pos)

End Sub
Private Function GetSheetScale() As Double

    Dim swSheet As SldWorks.Sheet
    Set swSheet = swDrawing.GetCurrentSheet
    
    Dim vProps  As Variant
    vProps = swSheet.GetProperties
    
    GetSheetScale = vProps(2) / vProps(3)
    
End Function
