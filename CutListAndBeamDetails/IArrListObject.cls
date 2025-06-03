VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "IArrListObject"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Private vItems As Variant
Private IsInit As Boolean

Property Get Items() As Variant
    
    Items = vItems

End Property

Property Get Count() As Long
    
    If IsEmpty(vItems) Then
        
        Count = 0
        
    Else
    
        Count = UBound(vItems) + 1
        
    End If

End Property


 Sub AddItems(InputItems As Variant)

    If Not IsEmpty(InputItems) Then
    
        If IsEmpty(vItems) Then
    
            vItems = InputItems
            
        Else
        
            Dim i As Integer
            For i = LBound(InputItems) To UBound(InputItems)
                
                Dim Obj As Object
                Set Obj = InputItems(i)
                
                Call AddtoList(Obj)

            Next i
            
        
        End If
        
    End If
    
End Sub


Private Sub Class_Initialize()

    IsInit = True

End Sub

Function AddtoList(Val As Object)

    If IsInit Then
        
        IsInit = False
        ReDim vItems(0)
        Set vItems(0) = Val
    
    Else
    
        ReDim Preserve vItems(UBound(vItems) + 1)
        Set vItems(UBound(vItems)) = Val
    
    End If
    
End Function

Public Sub SortItems(SortParam As String, Optional SortDescending As Boolean = True)
    
    If Not IsEmpty(vItems) Then
    
        vItems = MergeSort(vItems, SortParam, SortDescending)
        
    End If

End Sub

Private Function MergeSort(TestArr As Variant, SortParam As String, SortDescending As Boolean)

    If UBound(TestArr) = 0 Then

        MergeSort = TestArr

    Else

        Dim Mid As Integer
        Mid = Int((UBound(TestArr) - LBound(TestArr) + 1) / 2)

        Dim LeftArr() As Object
        Dim RightArr() As Object

        Dim i As Integer
        For i = LBound(TestArr) To Mid - 1

            ReDim Preserve LeftArr(i)
            Set LeftArr(UBound(LeftArr)) = TestArr(i)

        Next i

        Dim j As Integer
        For j = Mid To UBound(TestArr)

            ReDim Preserve RightArr(j - Mid)
            Set RightArr(UBound(RightArr)) = TestArr(j)

        Next j

        LeftArr = MergeSort(LeftArr, SortParam, SortDescending)
        RightArr = MergeSort(RightArr, SortParam, SortDescending)

        MergeSort = MergeArray(TestArr, RightArr, LeftArr, SortParam, SortDescending)

    End If

End Function

Private Function MergeArray(TotArr As Variant, RightArr As Variant, LeftArr As Variant, SortParam As String, SortDescending As Boolean) As Variant

    Dim i As Integer
    Dim j As Integer
    Dim k As Integer

    i = 0
    j = 0
    k = 0

    While (i <= UBound(LeftArr) And j <= UBound(RightArr))

        If CompareValues(CallByName(LeftArr(i), SortParam, VbGet), CallByName(RightArr(j), SortParam, VbGet), SortDescending) Then

            Set TotArr(k) = LeftArr(i)
            k = k + 1
            i = i + 1

        Else

            Set TotArr(k) = RightArr(j)
            k = k + 1
            j = j + 1

        End If

    Wend

    While (i <= UBound(LeftArr))

        Set TotArr(k) = LeftArr(i)
        k = k + 1
        i = i + 1

    Wend

    While (j <= UBound(RightArr))

        Set TotArr(k) = RightArr(j)
        k = k + 1
        j = j + 1

    Wend

    MergeArray = TotArr

End Function

Function CompareValues(Val1 As Variant, Val2 As Variant, SortDescending As Boolean) As Boolean
    
    Dim Result As Boolean
    
    If SortDescending Then
        
        Result = (Val1 >= Val2)
        
    Else
    
        Result = (Val1 <= Val2)
        
    End If
    
    CompareValues = Result

End Function

Function Clone() As IArrListObject

    Dim CloneList As IArrListObject
    Set CloneList = New IArrListObject
    
    If Not IsEmpty(vItems) Then
    
        For i = LBound(vItems) To UBound(vItems)
            
            Dim Obj As Object
            Set Obj = vItems(i)
            
            CloneList.AddtoList Obj
    
        Next i
        
    End If
        
    Set Clone = CloneList

End Function

Sub Reverse()
    
    If Not IsEmpty(vItems) Then
    
        Dim ReverseItems As Variant
        ReDim ReverseItems(UBound(vItems))
        
        Dim iter As Integer
        iter = LBound(vItems)
        
        Dim i As Integer
        
        For i = UBound(vItems) To LBound(vItems) Step -1
            
            ReverseItems(iter) = vItems(i)
            iter = iter + 1
            
        Next i
        
        vItems = ReverseItems
        
    End If
    
End Sub



