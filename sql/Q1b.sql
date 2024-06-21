SELECT
    CASE
        WHEN COUNT(1) > 0 THEN 'Yes'
        ELSE 'No'
    END
FROM
    tblInVitroSelection
WHERE
    RefID IN (
        SELECT RefID
        FROM tblReferences
        WHERE MedlineID = {pubmed_id}
    )
    AND
    SampleType = 'Clinical'
;
