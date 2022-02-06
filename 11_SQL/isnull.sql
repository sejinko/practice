CREATE TABLE CONTACTS
(
IDINT GENERATED BY DEFAULT AS IDENTITY
,FIRST_NAMEVARCHAR(50)NOT NULL
,LAST_NAMEVARCHAR(50)NOT NULL
,EMAILVARCHAR(255)NOT NULL
,PHONEVARCHAR(15)
,PRIMARY KEY (ID)
);

INSERT
INTO
CONTACTS(FIRST_NAME,LAST_NAME,EMAIL,PHONE)
VALUES
('John','Doe','john.doe@example.com',NULL),
('Lily','Bush','lily.bush@example.com'
,'(408-234-
2764)');

select
	*
from 
	contacts;
	

select 
	id,
	first_name,
	last_name,
	email,
	phone
from
	contacts
where phone isnull;


select 
	id,
	first_name,
	last_name,
	email,
	phone
from
	contacts
where phone is null;


select 
	id,
	first_name,
	last_name,
	email,
	phone
from
	contacts
where 
	phone is not null;