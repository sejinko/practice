CREATE TABLE EMPLOYEE
(
EMPLOYEE_ID INT PRIMARY KEY
, FIRST_NAME VARCHAR (255) NOT NULL
, LAST_NAME VARCHAR (255) NOT NULL
, MANAGER_ID INT
, FOREIGN KEY (MANAGER_ID)
REFERENCES EMPLOYEE (EMPLOYEE_ID)
ON DELETE CASCADE
);


INSERT INTO EMPLOYEE (
EMPLOYEE_ID
, FIRST_NAME
, LAST_NAME
, MANAGER_ID
)
VALUES
(1, 'Windy', 'Hays', NULL),
(2, 'Ava', 'Christensen', 1),
(3, 'Hassan', 'Conner', 1),
(4, 'Anna', 'Reeves', 2),
(5, 'Sau', 'Norman', 2),
(6, 'Kelsie', 'Hays', 3),
(7, 'Tory', 'Goff', 3),
(8, 'Salley', 'Lester', 3);


select * from employee;

select 
	e.first_name || ' ' || e.last_name employee,
	m.first_name || ' ' || m.last_name manager
from
	employee e
inner join employee m
on m.employee_id = e.manager_id 
order by manager;


select 
	e.first_name || ' ' || e.last_name employee,
	m.first_name || ' ' || m.last_name manager
from
	employee e
left join employee m 
on m.employee_id = e.manager_id 
order by manager asc;


select 
	f1.title,
	f2.title,
	f1.length
from
	film f1
inner join film f2
on f1.film_id != f2.film_id 
and f1.length = f2.length;