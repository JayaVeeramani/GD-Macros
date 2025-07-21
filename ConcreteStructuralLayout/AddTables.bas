Attribute VB_Name = "AddTables"
Sub InsertBOMAndOrderComponents(swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, ViewMaxLoc As Double)
    
    Dim swConfig As SldWorks.Configuration
    Set swConfig = swTopLevelModel.GetActiveConfiguration
    
    Dim configName As String
    configName = swConfig.Name
        
    Dim swBomTableAnnotation As SldWorks.BomTableAnnotation
    Set swBomTableAnnotation = swView.InsertBomTable2(False, 0.41595679, 0.27030866, swBOMConfigurationAnchorType_e.swBOMConfigurationAnchor_TopRight, _
                    swBomType_e.swBomType_PartsOnly, configName, "C:\FBD\COMMON\FBD Templates\SIMPLE BOM.sldbomtbt")
                    
    
    If Not swBomTableAnnotation Is Nothing Then
    
        Dim swTableAnn As SldWorks.TableAnnotation
        Set swTableAnn = swBomTableAnnotation
        
        Call OrderBOMTable(swTableAnn)
        Call SplitTableIfNeeded(swTableAnn, ViewMaxLoc)
    
    End If
    
End Sub

Private Sub OrderBOMTable(swTableAnn As SldWorks.TableAnnotation)
    
    Dim i As Integer
    For i = 1 To swTableAnn.rowCount - 1
    
        Dim Desc As String
        Desc = swTableAnn.DisplayedText(i, 2)
        
        Select Case True
        
            Case InStr(Desc, "WIRE MESH") > 0
            
                Call MoveTableRow(swTableAnn, i, 1)
                
            Case InStr(Desc, "#3") > 0 And InStr(Desc, "REBAR") And Not (InStr(Desc, "BEND") > 0)
                
                Call MoveTableRow(swTableAnn, i, 3)
                
            Case InStr(Desc, "#4") > 0 And InStr(Desc, "REBAR") > 0
            
                Call MoveTableRow(swTableAnn, i, 4)
            
            Case InStr(Desc, "#5") > 0 And InStr(Desc, "REBAR") > 0

                Call MoveTableRow(swTableAnn, i, 5)
            
            Case InStr(Desc, "#6") > 0 And InStr(Desc, "REBAR") > 0
                
                Call MoveTableRow(swTableAnn, i, 6)
            
            Case InStr(Desc, "FOAM") > 0
            
                Call MoveTableRow(swTableAnn, i, 7)

        
        End Select

    Next i
    
End Sub

Private Sub MoveTableRow(swTableAnn As SldWorks.TableAnnotation, ByRef CurRow As Integer, DestRow As Integer)

    If swTableAnn.rowCount - 1 >= DestRow Then
    
        If Not (DestRow = CurRow) Then
        
            Dim Bool As Boolean
            
            
            If DestRow > CurRow Then
            
                Bool = swTableAnn.MoveRow(CurRow, swTableItemInsertPosition_e.swTableItemInsertPosition_After, DestRow)
                CurRow = CurRow - 1
                
            ElseIf DestRow < CurRow Then
            
                Bool = swTableAnn.MoveRow(CurRow, swTableItemInsertPosition_e.swTableItemInsertPosition_Before, DestRow)
                Bool = swTableAnn.MoveRow(DestRow + 1, swTableItemInsertPosition_e.swTableItemInsertPosition_After, CurRow)
    
                
            End If
            
        End If

    End If
    

End Sub



Private Sub SplitTableIfNeeded(swTableAnn As SldWorks.TableAnnotation, ViewMaxLoc As Double)
    

    Dim TableWidth As Double
    TableWidth = GetColumnWidth(swTableAnn)

    Dim rowHeight As Double
    rowHeight = swTableAnn.GetRowHeight(0)
    Debug.Print swTableAnn.Text(1, 2)
    
    Dim ViewTopGap As Double
    ViewTopGap = SheetBorderTop - ViewMaxLoc - 0.01
    
    
    Dim i As Integer
    Dim NoOfRows As Integer
    NoOfRows = Int(ViewTopGap / rowHeight)
        
    Dim MaxNoOfSplits As Integer
    MaxNoOfSplits = Int((0.41595679 - 0.01590679) / TableWidth)
        
    If Int(swTableAnn.rowCount / NoOfRows) < MaxNoOfSplits Then
            
        MaxNoOfSplits = Int(swTableAnn.rowCount / NoOfRows)
            
    Else
            
        NoOfRows = Int(swTableAnn.rowCount / (MaxNoOfSplits + 1)) + 1
            
    End If
        
    If Abs(swTableAnn.rowCount - NoOfRows) > 2 Then
        
        For i = 1 To MaxNoOfSplits
    
            Set swTableAnn = swTableAnn.Split(swTableSplitLocations_e.swTableSplit_AfterRow, i * (NoOfRows - 1))
                    
            If Not swTableAnn Is Nothing Then
                    
                Dim swAnn As SldWorks.Annotation
                Set swAnn = swTableAnn.GetAnnotation()
                        
                swAnn.SetPosition2 0.41595679 - i * (TableWidth + 0.005), SheetBorderTop, 0
                        
            End If
 
        Next i
            
    End If

End Sub

Private Function GetColumnWidth(swTable As SldWorks.TableAnnotation) As Double
    
    Dim i As Integer
    For i = 0 To swTable.ColumnCount - 1
                
        GetColumnWidth = GetColumnWidth + swTable.GetColumnWidth(i)
        
    Next i

End Function

