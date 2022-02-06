create table cross_t1
(
	label char(1) primary key
);

create table cross_t2
(
	score int primary key
);


insert into cross_t1 (label)
values
('a'),
('b');

insert into cross_t2 (score)
values
(1),
(2),
(3);


select * from cross_t1;

select * from cross_t2;


select 
*
from 
cross_t1 
cross join
cross_t2;